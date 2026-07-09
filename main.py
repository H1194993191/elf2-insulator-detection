#!/usr/bin/env python3
"""
ELF2 绝缘子缺陷检测系统 (单文件版)
===================================
基于 RK3588 NPU + OV13855 摄像头 + MIPI 触摸屏的边缘计算方案。

功能:
  - 摄像头实时采集 (OV13855 MIPI-CSI /dev/video11)
  - RKNN NPU 推理 (5分类: 正常/破损/污秽/闪络痕迹/异物)
  - Qt 触屏界面 (1024x600, Wayland)
  - 自动截图 / 录像 / 事件日志
  - 检测到绝缘子自动触发 LLM 核验 (智谱 GLM-4V)
  - 核验完成自动保存报告 (JSON)
  - 历史记录列表, 点击查看详情

用法:
  # 板端默认启动 (摄像头 + NPU + Qt 界面)
  python3 main.py

  # 指定参数
  python3 main.py --camera /dev/video11 --model model.rknn --conf 0.25

  # 开启录像
  python3 main.py --record

  # 启用 LLM 核验 (缺陷自动核验)
  python3 main.py --verify-api zhipu

环境要求:
  - RK3588 板端, Python 3.10
  - rknn-toolkit-lite2, PyQt5, OpenCV, numpy
  - Wayland 显示服务 (板端默认)
"""

import argparse
import base64
import glob
import json
import os
import random
import signal
import sys
import time
import threading
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

import cv2
import numpy as np

# ---- Web 仪表盘 ----
from web_server import WebServer

# 提前导入 PyQt5 (模块级引用)
try:
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel,
        QPushButton, QVBoxLayout, QHBoxLayout, QGridLayout, QFrame, QDialog,
        QTextEdit, QTableWidget, QTableWidgetItem, QHeaderView, QStackedWidget,
        QProgressBar, QGroupBox, QSplitter, QListWidget, QListWidgetItem,
        QScrollArea)
    from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
    from PyQt5.QtGui import QImage, QPixmap, QFont, QColor, QPalette
    _HAS_PYQT5 = True
except ImportError:
    _HAS_PYQT5 = False

# ============================================================
# 全局常量
# ============================================================
CLASS_NAMES = {
    0: "Nora1", 1: "JYZ", 2: "Broken",
}
CLASS_CN = {
    0: "正常", 1: "绝缘子", 2: "破损",
}
CLASS_COLORS_BGR = {
    0: (0, 255, 0),       # 正常 - 绿色
    1: (255, 255, 0),     # 绝缘子 - 青色(检测目标,非缺陷)
    2: (0, 0, 255),       # 破损 - 红色(唯一缺陷类)
}
CLASS_COLORS_HEX = {
    0: "#4caf50", 1: "#00bcd4", 2: "#f44336",
}

SCREEN_W, SCREEN_H = 960, 600
VIDEO_W, VIDEO_H = 700, 460
PANEL_W = 240

# 自动核验冷却时间 (秒), 防止同一缺陷反复触发
VERIFY_COOLDOWN = 2.0


# ============================================================
# 摄像头采集
# ============================================================
class CameraCapture:
    """OV13855 摄像头采集, GStreamer 优先, V4L2 回退"""

    def __init__(self, source: str = "/dev/video11", width: int = 1920,
                 height: int = 1080, fps: int = 15, use_gstreamer: bool = True):
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.use_gstreamer = use_gstreamer
        self.cap = None

    def open(self) -> bool:
        if self.use_gstreamer:
            pipeline = (
                f"v4l2src device={self.source} ! "
                f"video/x-raw,format=NV12,width={self.width},height={self.height},"
                f"framerate={self.fps}/1 ! "
                f"videoconvert ! video/x-raw,format=BGR ! appsink drop=1 max-buffers=2"
            )
            try:
                self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
                if self.cap.isOpened():
                    print(f"[INFO] GStreamer 摄像头已打开: {self.source}")
                    return True
            except Exception as e:
                print(f"[WARN] GStreamer 失败: {e}")
        return self._open_v4l2()

    def _open_v4l2(self) -> bool:
        try:
            src = int(self.source.replace("/dev/video", ""))
        except ValueError:
            src = self.source
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            print(f"[ERROR] 无法打开摄像头: {self.source}")
            return False
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        print(f"[INFO] V4L2 摄像头已打开: {self.source}")
        return True

    def read(self):
        if self.cap is None or not self.cap.isOpened():
            return False, None
        return self.cap.read()

    def release(self):
        if self.cap:
            self.cap.release()
            self.cap = None


# ============================================================
# RKNN 检测器
# ============================================================
class RKNNDetector:
    """RKNN NPU 推理 + NMS 后处理"""

    def __init__(self, model_path: str, conf_thres: float = 0.25,
                 iou_thres: float = 0.45, input_size: int = 640,
                 defect_conf_thres: float = 0.25):
        self.model_path = model_path
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.defect_conf_thres = defect_conf_thres  # 缺陷类专用阈值
        self.input_size = input_size
        self._dec_frame = 0  # 调试日志帧计数
        self.rknn = None
        self._available = False
        # CLAHE 预处理: 提升摄像头画面局部对比度, 缩小与数据集画质差距
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def load(self) -> bool:
        try:
            from rknnlite.api import RKNNLite
            self.rknn = RKNNLite()
            if self.rknn.load_rknn(self.model_path) != 0:
                print(f"[ERROR] RKNN 模型加载失败: {self.model_path}")
                return False
            if self.rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2) != 0:
                print("[ERROR] RKNN 运行时初始化失败")
                return False
            self._available = True
            print(f"[INFO] RKNN 模型已加载: {self.model_path}")
            return True
        except ImportError:
            print("[WARN] rknnlite 不可用, NPU 推理禁用")
            return True
        except Exception as e:
            print(f"[ERROR] RKNN 初始化异常: {e}")
            return False

    def is_available(self) -> bool:
        return self._available

    def detect(self, frame: np.ndarray) -> List[List[float]]:
        if not self._available:
            return []
        h0, w0 = frame.shape[:2]
        # CLAHE 预处理: 提升局部对比度, 拉近摄像头与数据集画质差距, 减少误检
        try:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = self._clahe.apply(l)
            frame = cv2.merge([l, a, b])
            frame = cv2.cvtColor(frame, cv2.COLOR_LAB2BGR)
        except Exception:
            pass  # CLAHE 失败不影响主流程
        inp, r, pad = self._letterbox(frame, (self.input_size, self.input_size))
        inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
        inp = np.expand_dims(inp, axis=0)
        try:
            outputs = self.rknn.inference(inputs=[inp])
            return self._postprocess(outputs, w0, h0, r, pad)
        except Exception:
            return []

    def _letterbox(self, img, new_shape):
        shape = img.shape[:2]
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        nu = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw, dh = (new_shape[1] - nu[0]) / 2, (new_shape[0] - nu[1]) / 2
        if shape[::-1] != nu:
            img = cv2.resize(img, nu, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        return cv2.copyMakeBorder(img, top, bottom, left, right,
                                   cv2.BORDER_CONSTANT, value=(114, 114, 114)), r, (dw, dh)

    def _postprocess(self, outputs, ow, oh, r, pad):
        tensors = [np.asarray(o) for o in outputs]
        tensors.sort(key=lambda x: x.size, reverse=True)
        for t in tensors:
            dets = self._decode(t, ow, oh, r, pad)
            if dets:
                return dets
        return []

    def _decode(self, t, ow, oh, r, pad):
        if t.ndim == 3:
            t = t[0]
        if t.shape[0] < t.shape[1]:
            t = t.transpose(1, 0)
        if t.shape[1] < 6:
            return []
        bb = t[:, :4]
        cs = t[:, 4:]
        cid = np.argmax(cs, axis=1)
        cc = cs[np.arange(len(cs)), cid]
        mask = cc >= self.conf_thres
        if not np.any(mask):
            return []
        bb, cid, cc, cs_full = bb[mask], cid[mask], cc[mask], cs[mask]
        # 兼容归一化坐标 (0~1) 和 640 空间坐标
        if np.max(bb) < 2.0:
            bb = bb * self.input_size
        x, y, w, h = bb[:, 0], bb[:, 1], bb[:, 2], bb[:, 3]
        dw, dh = pad
        # NPU 输出的是 640x640 letterbox 空间坐标，转换回原始图像坐标
        x1 = np.clip((x - w / 2 - dw) / r, 0, ow - 1)
        y1 = np.clip((y - h / 2 - dh) / r, 0, oh - 1)
        x2 = np.clip((x + w / 2 - dw) / r, 0, ow - 1)
        y2 = np.clip((y + h / 2 - dh) / r, 0, oh - 1)
        boxes = np.stack([x1, y1, x2, y2], axis=1)
        keep = self._nms(boxes, cc)
        dets = [[float(boxes[i, 0]), float(boxes[i, 1]), float(boxes[i, 2]),
                 float(boxes[i, 3]), float(cc[i]), int(cid[i])] for i in keep]

        # 对保留的每个检测框，检查其重叠区域内是否有 class 2(破损) 高响应
        # (class 2 分数通常低于 class 1，会被 NMS 抑制，这里做补救)
        # BRK_THRES 联动 --defect-conf 参数, 默认 0.25 远高于旧版 0.10, 减少误检
        BRK_THRES = self.defect_conf_thres
        if cs_full.shape[1] > 2:
            for i_k in range(len(keep)):
                i = keep[i_k]
                x1_i, y1_i, x2_i, y2_i = boxes[i]
                area_i = (x2_i - x1_i) * (y2_i - y1_i) + 1e-6
                max_brk = 0.0
                for j in range(len(boxes)):
                    x1_j, y1_j, x2_j, y2_j = boxes[j]
                    xx1 = max(x1_i, x1_j); yy1 = max(y1_i, y1_j)
                    xx2 = min(x2_i, x2_j); yy2 = min(y2_i, y2_j)
                    inter = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
                    union = area_i + (x2_j - x1_j) * (y2_j - y1_j) - inter + 1e-6
                    iou = inter / union
                    if iou >= 0.3 and float(cs_full[j, 2]) > max_brk:
                        max_brk = float(cs_full[j, 2])
                if max_brk >= BRK_THRES and int(dets[i_k][5]) != 2:
                    dets[i_k][5] = 2
                    dets[i_k][4] = max_brk
        return dets

    @staticmethod
    def _nms(boxes, scores):
        x1, y1 = boxes[:, 0], boxes[:, 1]
        x2, y2 = boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            order = order[1:][iou <= 0.45]
        return keep

    def release(self):
        if self.rknn:
            self.rknn.release()
            self.rknn = None


# ============================================================
# LLM 智能核验
# ============================================================
class LLMVerifier:
    """大模型核验: 智谱 GLM-4V 多模态视觉核验"""

    def __init__(self):
        self._config: Dict[str, str] = {}
        self._enabled = False

    def configure(self, api_key: str = "", api_url: str = "", model: str = ""):
        self._config = {"api_key": api_key, "api_url": api_url, "model": model}
        self._enabled = bool(api_key)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def verify(self, frame: np.ndarray, detections: List) -> Dict:
        """调用 LLM 核验检测结果, 返回结构化 JSON

        detections 已经由 _strip_big_enclosing_box() 过滤掉大框,
        此处直接对每个独立绝缘子逐一核验。"""
        if not self._enabled or not detections:
            return {"verifications": [], "overall_assessment": ""}

        # 提取图像特征
        img_features = self._extract_image_features(frame)

        # 提取每个检测框的 ROI 特征
        roi_dets = []
        for d in detections:
            x1, y1, x2, y2 = int(d[0]), int(d[1]), int(d[2]), int(d[3])
            crop = frame[max(0, y1):y2, max(0, x1):x2]
            roi_dets.append({
                "class_id": int(d[5]),
                "class_name": CLASS_CN.get(int(d[5]), str(int(d[5]))),
                "confidence": round(float(d[4]), 4),
                "bbox": [x1, y1, x2, y2],
                "bbox_area": int((x2 - x1) * (y2 - y1)),
                "roi_features": self._extract_roi_features(crop),
            })

        try:
            prompt = self._build_prompt(roi_dets, img_features)
            img_b64 = ""
            if "bigmodel" in self._config.get("api_url", ""):
                img_b64 = self._encode_frame_for_vision(frame, detections)
            result = self._call_api(prompt, img_b64)

            # 根据逐项结果汇总整体评估
            vers = result.get("verifications", [])
            has_defect = any(
                v.get("severity", "none") in ("moderate", "severe", "critical")
                for v in vers
            ) or any(int(d[5]) == 2 for d in detections)

            llm_summary = result.get("overall_assessment", "")
            result["overall_assessment"] = (
                f"检测到{len(detections)}个绝缘子, "
                + ("存在破损缺陷。" if has_defect else "状态良好。")
                + llm_summary
            )
            return result
        except Exception as e:
            print(f"[WARN] LLM 核验失败: {e}")
            return {"verifications": [], "overall_assessment": str(e)}

    def _encode_frame_for_vision(self, frame, detections):
        """在帧上画检测框后编码为 base64, 供视觉模型分析"""
        import base64
        vis = frame.copy()
        for d in detections:
            x1, y1, x2, y2 = int(d[0]), int(d[1]), int(d[2]), int(d[3])
            cls = int(d[5])
            color = CLASS_COLORS_BGR.get(cls, (0, 255, 0))
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            label = f"{CLASS_NAMES.get(cls, str(cls))} {d[4]:.2f}"
            cv2.putText(vis, label, (x1, max(y1 - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        # 缩小图片以节省 token (最长边不超过 1024)
        h, w = vis.shape[:2]
        if max(h, w) > 1024:
            scale = 1024 / max(h, w)
            vis = cv2.resize(vis, (int(w * scale), int(h * scale)))
        _, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 75])
        return base64.b64encode(buf).decode("utf-8")

    def _extract_image_features(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return {
            "resolution": f"{img.shape[1]}x{img.shape[0]}",
            "mean_brightness": round(float(np.mean(gray)), 1),
            "std_brightness": round(float(np.std(gray)), 1),
        }

    def _extract_roi_features(self, crop):
        if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
            return {}
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (crop.shape[0] * crop.shape[1])
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return {
            "roi_size": f"{crop.shape[1]}x{crop.shape[0]}",
            "mean_brightness": round(float(np.mean(gray)), 1),
            "edge_density": round(float(edge_density), 4),
            "laplacian_variance": round(float(laplacian_var), 1),
        }

    def _build_prompt(self, detections, img_features):
        """构建核验 Prompt, 逐一核验每个绝缘子目标"""
        det_lines = []
        for i, det in enumerate(detections):
            rf = det.get("roi_features", {})
            det_lines.append(
                f"目标#{i+1}: {det['class_name']} | 置信度:{det['confidence']:.2%} | "
                f"边缘密度:{rf.get('edge_density','N/A')} | "
                f"纹理方差:{rf.get('laplacian_variance','N/A')}"
            )
        return (
            f"你是电力巡检专家,请逐一核验以下{len(detections)}个绝缘子目标。\n\n"
            f"【图像信息】分辨率:{img_features['resolution']}, "
            f"亮度:{img_features['mean_brightness']}±{img_features['std_brightness']}\n"
            f"【检测结果】共{len(detections)}个独立绝缘子:\n"
            + "\n".join(det_lines) +
            "\n\n严格按以下JSON格式返回(只返回JSON,不要任何其他文字):\n"
            "{\n"
            "  \"verifications\": [\n"
            '    {"target_index":1,"verification":"correct|incorrect|uncertain",'
            '"severity":"none|minor|moderate|severe|critical",'
            '"defect_description":"...","maintenance_suggestion":"..."},\n'
            "    ...\n"
            "  ],\n"
            '  "overall_assessment": "整体评估总结"\n'
            "}\n"
            "\n核验规则:\n"
            f"1. target_index从1到{len(detections)},每个目标必须逐一核验,不可跳过!\n"
            "2. 绝缘子完好→verification:correct+severity:none,描述\"绝缘子外观完好\"\n"
            "3. 绝缘子破损/裂纹→verification:correct+severity:severe,描述具体缺陷位置和形态\n"
            "4. 误检(不是绝缘子)→verification:incorrect+severity:none\n"
            "5. 存在破损时maintenance_suggestion必须写明更换/维修建议,严禁写\"无需维护\"\n"
            "6. overall_assessment整合所有目标的判断,简明扼要。"
        )

    def _call_api(self, prompt, image_b64: str = ""):
        import requests
        headers = {
            "Authorization": f"Bearer {self._config['api_key']}",
            "Content-Type": "application/json",
        }
        # 智谱 GLM-4V 支持图片输入, 发送 base64 图片让模型真正"看到"绝缘子
        if image_b64 and "bigmodel" in self._config.get("api_url", ""):
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                    {"type": "text", "text": prompt},
                ]
            }]
        else:
            messages = [{"role": "user", "content": prompt}]

        payload = {
            "model": self._config["model"],
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 2048,
        }

        resp = requests.post(self._config["api_url"], headers=headers,
                             json=payload, timeout=(10, 60))
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]

        # 处理 LLM 返回的内容可能被 markdown 代码块包裹的情况
        content = content.strip()
        if content.startswith("```"):
            # 去掉 ```json ... ``` 包裹
            lines = content.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            content = "\n".join(lines)

        try:
            result = json.loads(content)
            # 兜底修复: LLM 可能把 overall_assessment 塞进 verifications 数组里
            vers = result.get("verifications", [])
            for i, v in enumerate(vers):
                if isinstance(v, dict) and "overall_assessment" in v:
                    result["overall_assessment"] = v["overall_assessment"]
                    vers.pop(i)
                    break
            result["verifications"] = [v for v in vers
                                       if isinstance(v, dict) and "target_index" in v]
            return result
        except json.JSONDecodeError as e:
            print(f"[核验警告] LLM返回无法解析为JSON: {e}")
            return {"verifications": [], "overall_assessment": content}


# ============================================================
# 大框识别: 模型偶尔检出包住整串绝缘子的包围框, 分离出来只用于显示
# ============================================================
def _classify_boxes(detections):
    """分离包围整串绝缘子的大框和独立绝缘子小框
    Returns: (small_dets, big_boxes)
        - small_dets: 独立绝缘子, 用于核验/统计
        - big_boxes: 包围框, 只用于画面显示
    """
    n = len(detections)
    if n <= 3:
        return detections, []  # 太少, 不可能有大框包小框

    # 收集各框面积
    boxes = []
    for i, d in enumerate(detections):
        x1, y1, x2, y2 = int(d[0]), int(d[1]), int(d[2]), int(d[3])
        area = (x2 - x1) * (y2 - y1)
        boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "area": area, "idx": i})

    median_area = sorted(b["area"] for b in boxes)[n // 2]

    big_idx = None
    for b in boxes:
        if b["area"] <= 4.0 * median_area:
            continue
        contained = 0
        for b2 in boxes:
            if b["idx"] == b2["idx"]:
                continue
            cx = (b2["x1"] + b2["x2"]) / 2.0
            cy = (b2["y1"] + b2["y2"]) / 2.0
            if b["x1"] <= cx <= b["x2"] and b["y1"] <= cy <= b["y2"]:
                contained += 1
        if contained >= 2:
            big_idx = b["idx"]
            break

    if big_idx is None:
        return detections, []

    big_box = detections[big_idx]
    small_dets = [detections[i] for i in range(n) if i != big_idx]
    return small_dets, [big_box]


# ============================================================
# 数据记录
# ============================================================
class DataRecorder:
    """截图 / 录像 / 事件日志"""

    def __init__(self, output_dir: str = "records", save_video: bool = False):
        self.output_dir = Path(output_dir)
        self.save_video = save_video
        self.snapshot_dir = self.output_dir / "snapshots"
        self.report_dir = self.output_dir / "reports"
        for d in [self.output_dir, self.snapshot_dir, self.report_dir]:
            d.mkdir(parents=True, exist_ok=True)
        self._writer = None
        self._session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._frame_count = 0
        self._events: List[Dict] = []

    def start_video(self, fps: float, frame_size: tuple):
        if not self.save_video:
            return
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        path = str(self.output_dir / f"video_{self._session_id}.mp4")
        self._writer = cv2.VideoWriter(path, fourcc, fps, frame_size)
        print(f"[INFO] 录像开始: {path}")

    def write_frame(self, frame: np.ndarray, detections: List):
        self._frame_count += 1
        if self._writer:
            self._writer.write(frame)
        for d in detections:
            cls = int(d[5])
            if cls != 0:
                self._events.append({
                    "frame": self._frame_count,
                    "timestamp": datetime.now().isoformat(),
                    "class_id": cls,
                    "class_name": CLASS_CN.get(cls, str(cls)),
                    "confidence": float(d[4]),
                    "bbox": [int(d[0]), int(d[1]), int(d[2]), int(d[3])],
                })

    def save_snapshot(self, frame: np.ndarray) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = str(self.snapshot_dir / f"snap_{ts}.jpg")
        cv2.imwrite(path, frame)
        return path

    def save_verify_report(self, detections: List, verify_result: Dict,
                           snapshot_path: str = "") -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        report = {
            "timestamp": ts,
            "snapshot_path": snapshot_path,
            "detections": [{"class": CLASS_CN.get(int(d[5]), "?"),
                            "confidence": float(d[4]),
                            "bbox": [int(d[0]), int(d[1]), int(d[2]), int(d[3])]}
                           for d in detections],
            "verifications": verify_result.get("verifications", []),
            "overall_assessment": verify_result.get("overall_assessment", ""),
        }
        path = str(self.report_dir / f"report_{ts}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        return path

    def save_events(self):
        if self._events:
            path = self.output_dir / f"events_{self._session_id}.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._events, f, indent=2, ensure_ascii=False)
            print(f"[INFO] 事件日志已保存: {path} ({len(self._events)}条)")

    def release(self):
        self.save_events()
        if self._writer:
            self._writer.release()
            self._writer = None


# ============================================================
# Qt 触屏界面
# ============================================================
def _init_qt():
    """初始化 Qt 模块, 检查 PyQt5 是否可用"""
    return _HAS_PYQT5


# ---- 检测线程 ----
class DetectThread(QThread):
    """后台线程: 摄像头采集 + NPU 推理"""
    frame_ready = pyqtSignal(np.ndarray, list, float)
    status_update = pyqtSignal(str)

    def __init__(self, args):
        super().__init__()
        self.args = args
        self._running = False
        self._paused = False
        self.capture: Optional[CameraCapture] = None
        self.detector: Optional[RKNNDetector] = None
        # 多帧投票: 连续3帧检测到破损才确认, 单帧误检不触发
        self._broken_streak = 0
        self._broken_vote_threshold = 3

    def run(self):
        self._running = True

        # ---- 测试模式: 从文件夹随机取5张图 ----
        if self.args.test_dir:
            self._run_test_mode()
            self._release()
            return

        # 初始化摄像头
        self.capture = CameraCapture(
            source=self.args.camera, width=self.args.width,
            height=self.args.height, fps=self.args.fps)
        if not self.capture.open():
            self.status_update.emit("摄像头打开失败!")
            return
        self.status_update.emit(f"摄像头: {self.args.camera}")

        # 初始化检测器
        self.detector = RKNNDetector(
            model_path=self.args.model, conf_thres=self.args.conf,
            iou_thres=self.args.iou, input_size=self.args.input_size,
            defect_conf_thres=self.args.defect_conf)
        if self.detector.load():
            self.status_update.emit("NPU: 就绪")
        else:
            self.status_update.emit("NPU: 加载失败")

        fps_window = deque(maxlen=30)

        while self._running:
            if self._paused:
                self.msleep(100)
                continue

            t0 = time.time()
            ret, frame = self.capture.read()
            if not ret or frame is None:
                self.msleep(10)
                continue

            # 推理
            detections = self.detector.detect(frame) if self.detector and self.detector.is_available() else []

            # 多帧投票: 单帧 noise 误检不通过, 需连续3帧确认才保留 class 2(破损)
            has_broken_this_frame = any(int(d[5]) == 2 for d in detections)
            if has_broken_this_frame:
                self._broken_streak += 1
            else:
                self._broken_streak = 0

            if self._broken_streak < self._broken_vote_threshold:
                # 未达确认阈值, 将 class 2 降级为 class 0(绝缘子-良好)
                for d in detections:
                    if int(d[5]) == 2:
                        d[5] = 0

            # 在帧上绘制检测框
            disp = self._draw_detections(frame.copy(), detections)

            # FPS
            fps = 1.0 / max(1e-6, time.time() - t0)
            fps_window.append(fps)
            avg_fps = sum(fps_window) / len(fps_window)

            self.frame_ready.emit(disp, detections, avg_fps)

        self._release()

    def _run_test_mode(self):
        """测试模式: 随机抽取5张图片逐张检测核验"""
        test_dir = self.args.test_dir
        exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
        images = []
        for ext in exts:
            images.extend(glob.glob(os.path.join(test_dir, "**", ext), recursive=True))
            images.extend(glob.glob(os.path.join(test_dir, "**", ext.upper()), recursive=True))
        # 排除 _test_results 等结果目录中的图片
        images = [p for p in images if "_test_results" not in p and "_results" not in p]
        if not images:
            self.status_update.emit(f"测试目录无图片: {test_dir}")
            return

        random.shuffle(images)
        images = images[:5]
        self.status_update.emit(f"测试模式: 从 {len(images)} 张图片开始检测...")

        # 初始化检测器
        self.detector = RKNNDetector(
            model_path=self.args.model, conf_thres=self.args.conf,
            iou_thres=self.args.iou, input_size=self.args.input_size,
            defect_conf_thres=self.args.defect_conf)
        if self.detector.load():
            self.status_update.emit("NPU: 就绪")
        else:
            self.status_update.emit("NPU: 加载失败")
            return

        for idx, img_path in enumerate(images):
            if not self._running:
                break

            frame = cv2.imread(img_path)
            if frame is None:
                self.status_update.emit(f"无法读取: {Path(img_path).name}")
                continue

            self.status_update.emit(f"[{idx+1}/5] {Path(img_path).name}")

            # NPU 推理
            t0 = time.time()
            detections = self.detector.detect(frame) if self.detector and self.detector.is_available() else []
            fps = 1.0 / max(1e-6, time.time() - t0)

            # 画框
            disp = self._draw_detections(frame.copy(), detections)

            # 发送到主线程 → 自动触发截图+核验+历史记录
            self.frame_ready.emit(disp, detections, fps)

            # 等 4 秒让用户看到画面 + 核验完成
            for _ in range(40):
                if not self._running:
                    break
                self.msleep(100)

        self.status_update.emit("测试完成, 请查看结果页面")

    def stop(self):
        self._running = False
        self.wait(3000)

    def toggle_pause(self) -> bool:
        self._paused = not self._paused
        return self._paused

    @property
    def paused(self) -> bool:
        return self._paused

    def _draw_detections(self, frame, detections):
        for d in detections:
            x1, y1, x2, y2, score, cls = int(d[0]), int(d[1]), int(d[2]), int(d[3]), d[4], int(d[5])
            color = CLASS_COLORS_BGR.get(cls, (128, 128, 128))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{CLASS_NAMES.get(cls, str(cls))} {score:.2f}"
            cv2.putText(frame, label, (x1, max(10, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame

    def _release(self):
        if self.capture:
            self.capture.release()
        if self.detector:
            self.detector.release()


# ---- 报告弹窗 ----
class ReportDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("核验报告详情")
        self.setMinimumSize(750, 450)
        self.setStyleSheet("background: #1e1e1e; color: #e0e0e0;")
        layout = QVBoxLayout(self)
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setStyleSheet(
            "QTextEdit{background:#252526;color:#d4d4d4;"
            "font-family:monospace;font-size:14px;border:1px solid #3c3c3c;}")
        layout.addWidget(self.text_edit)
        btn = QPushButton("关闭")
        btn.setStyleSheet(
            "QPushButton{background:#d32f2f;color:white;border:none;"
            "padding:12px 24px;font-size:16px;border-radius:6px;}"
            "QPushButton:hover{background:#b71c1c;}")
        btn.clicked.connect(self.close)
        layout.addWidget(btn)

    def set_text(self, text: str):
        self.text_edit.setPlainText(text)


# ---- 主窗口 ----
class MainWindow(QMainWindow):
    # 跨线程信号: 后台线程 -> 主线程, 替代 QTimer.singleShot
    _sig_verify_done = pyqtSignal(object, list, np.ndarray)
    _sig_verify_failed = pyqtSignal(str)

    def __init__(self, args):
        super().__init__()
        self.args = args

        # 连接跨线程核验信号
        self._sig_verify_done.connect(self._on_verify_safe)
        self._sig_verify_failed.connect(self._on_verify_failed)
        self._det_thread: Optional[DetectThread] = None
        self._current_frame: Optional[np.ndarray] = None
        self._current_dets: List = []
        self._current_fps = 0.0

        # 历史记录列表
        self._history: List[Dict] = []

        # 自动核验状态
        self._verify_busy = False           # 是否正在核验中
        self._last_verify_time = 0.0        # 上次核验触发时间
        self._auto_snap_enabled = True      # 自动截图开关
        self._auto_verify_enabled = True    # 自动核验开关
        self._pending_verify = False        # 是否有待处理的核验请求
        self._pending_frame = None
        self._pending_dets = None

        # 数据记录器
        self.recorder = DataRecorder(
            output_dir=args.output_dir, save_video=args.record)

        # LLM 核验器
        self.verifier: Optional[LLMVerifier] = None
        self._init_verifier()

        self._init_ui()
        self._start_detection()

        # ---- Web 仪表盘 ----
        self.web_server: Optional[WebServer] = None
        if getattr(args, "web", False):
            try:
                self.web_server = WebServer(host="0.0.0.0", port=args.web_port)
                self.web_server._history = self._history  # 共享历史记录引用
                self.web_server.start()
            except Exception as e:
                print(f"[WEB] 启动失败: {e}")

    def _init_verifier(self):
        api = self.args.verify_api
        if not api:
            return
        self.verifier = LLMVerifier()
        if api == "zhipu":
            key = os.environ.get("ZHIPU_API_KEY", "")
            self.verifier.configure(
                api_key=key,
                api_url="https://open.bigmodel.cn/api/paas/v4/chat/completions",
                model="glm-4v")

    # ============================================================
    # UI 初始化
    # ============================================================
    def _init_ui(self):
        self.setWindowTitle("ELK 3588 绝缘子智能巡检系统")
        self.setGeometry(60, 0, SCREEN_W, SCREEN_H)
        self.setFixedSize(SCREEN_W, SCREEN_H)
        self.setStyleSheet("background: #1a1a2e;")

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.page_live = QWidget()
        self.page_result = QWidget()
        self.stack.addWidget(self.page_live)
        self.stack.addWidget(self.page_result)

        self._init_live_page()
        self._init_result_page()

        # 时钟
        self._time_timer = QTimer()
        self._time_timer.timeout.connect(self._update_time)
        self._time_timer.start(1000)

        # 录像状态
        self._recording = self.args.record

    def _init_live_page(self):
        layout = QHBoxLayout(self.page_live)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # ---- 左侧视频区域 ----
        video_frame = QFrame()
        video_frame.setStyleSheet(
            "background:#000;border:2px solid #16213e;border-radius:6px;")
        video_frame.setFixedSize(VIDEO_W, VIDEO_H)
        video_layout = QVBoxLayout(video_frame)
        video_layout.setContentsMargins(0, 0, 0, 0)

        self.video_label = QLabel("等待视频流...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet(
            "color:#555;font-size:18px;background:transparent;border:none;")
        video_layout.addWidget(self.video_label)
        layout.addWidget(video_frame, alignment=Qt.AlignTop)

        # ---- 右侧控制面板 ----
        panel = QFrame()
        panel.setStyleSheet("background:#16213e;border-radius:6px;")
        panel.setFixedSize(PANEL_W, VIDEO_H)
        pl = QVBoxLayout(panel)
        pl.setContentsMargins(10, 8, 10, 8)
        pl.setSpacing(4)

        # 标题
        t = QLabel("控制面板")
        t.setAlignment(Qt.AlignCenter)
        t.setStyleSheet(
            "color:#e94560;font-size:16px;font-weight:bold;background:transparent;")
        pl.addWidget(t)

        sep = QFrame(); sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color:#0f3460;"); pl.addWidget(sep)

        self.lbl_status = QLabel("状态: 初始化...")
        self.lbl_status.setStyleSheet("color:#a0a0a0;font-size:11px;background:transparent;")
        self.lbl_status.setWordWrap(True)
        pl.addWidget(self.lbl_status)

        self.lbl_fps = QLabel("FPS: --")
        self.lbl_fps.setStyleSheet("color:#00ff88;font-size:13px;font-weight:bold;background:transparent;")
        pl.addWidget(self.lbl_fps)

        self.lbl_stats = QLabel("检测: --")
        self.lbl_stats.setStyleSheet("color:#e0e0e0;font-size:11px;background:transparent;")
        self.lbl_stats.setWordWrap(True)
        pl.addWidget(self.lbl_stats)

        # 核验状态指示
        self.lbl_verify_status = QLabel("")
        self.lbl_verify_status.setStyleSheet("color:#ff9800;font-size:11px;background:transparent;")
        self.lbl_verify_status.setWordWrap(True)
        pl.addWidget(self.lbl_verify_status)

        sep2 = QFrame(); sep2.setFrameShape(QFrame.HLine)
        sep2.setStyleSheet("color:#0f3460;"); pl.addWidget(sep2)

        # 按钮
        btn_style_small = (
            "QPushButton{{background:{bg};color:white;border:none;"
            "padding:6px;font-size:12px;border-radius:5px;}}"
            "QPushButton:hover{{background:{hover};}}")
        self.btn_pause = QPushButton("⏸ 暂停")
        self.btn_pause.setStyleSheet(btn_style_small.format(bg="#4caf50", hover="#388e3c"))
        self.btn_pause.clicked.connect(self._toggle_pause)
        self.btn_pause.setMinimumHeight(34)
        pl.addWidget(self.btn_pause)

        self.btn_snap = QPushButton("📷 截图")
        self.btn_snap.setStyleSheet(btn_style_small.format(bg="#2196f3", hover="#1565c0"))
        self.btn_snap.clicked.connect(self._snapshot)
        self.btn_snap.setMinimumHeight(34)
        pl.addWidget(self.btn_snap)

        self.btn_record = QPushButton("⏺ 录像")
        self.btn_record.setStyleSheet(btn_style_small.format(bg="#607d8b", hover="#455a64"))
        self.btn_record.clicked.connect(self._toggle_record)
        self.btn_record.setMinimumHeight(34)
        pl.addWidget(self.btn_record)

        self.btn_auto_toggle = QPushButton("🔍 自动核验: ON")
        self.btn_auto_toggle.setStyleSheet(btn_style_small.format(bg="#4caf50", hover="#388e3c"))
        self.btn_auto_toggle.clicked.connect(self._toggle_auto_verify)
        self.btn_auto_toggle.setMinimumHeight(34)
        pl.addWidget(self.btn_auto_toggle)

        self.btn_result = QPushButton("📊 历史记录")
        self.btn_result.setStyleSheet(btn_style_small.format(bg="#9c27b0", hover="#6a1b9a"))
        self.btn_result.clicked.connect(self._show_result_page)
        self.btn_result.setMinimumHeight(34)
        pl.addWidget(self.btn_result)

        self.btn_exit = QPushButton("✕ 退出")
        self.btn_exit.setStyleSheet(btn_style_small.format(bg="#f44336", hover="#b71c1c"))
        self.btn_exit.clicked.connect(self._exit)
        self.btn_exit.setMinimumHeight(34)
        pl.addWidget(self.btn_exit)

        pl.addStretch()

        self.lbl_time = QLabel()
        self.lbl_time.setAlignment(Qt.AlignCenter)
        self.lbl_time.setStyleSheet("color:#888;font-size:10px;background:transparent;")
        pl.addWidget(self.lbl_time)

        layout.addWidget(panel, alignment=Qt.AlignTop)
        layout.addStretch()

    def _init_result_page(self):
        page = self.page_result
        page.setStyleSheet("background: #1a1a2e;")
        main_layout = QVBoxLayout(page)
        main_layout.setContentsMargins(8, 6, 8, 6)
        main_layout.setSpacing(4)

        # ---- 标题栏（含返回按钮，确保始终可见） ----
        title_bar = QHBoxLayout()
        title_bar.setSpacing(8)

        self.btn_back = QPushButton("⬅ 返回")
        self.btn_back.setStyleSheet(
            "QPushButton{background:#2196f3;color:white;border:none;"
            "padding:6px 14px;font-size:12px;border-radius:5px;}"
            "QPushButton:hover{background:#1565c0;}")
        self.btn_back.clicked.connect(self._show_live_page)
        self.btn_back.setFixedSize(80, 34)
        title_bar.addWidget(self.btn_back)

        title = QLabel("检测历史记录")
        title.setStyleSheet(
            "color:#e94560;font-size:18px;font-weight:bold;background:transparent;")
        title_bar.addWidget(title)
        title_bar.addStretch()

        self.lbl_history_count = QLabel("共 0 条记录")
        self.lbl_history_count.setStyleSheet(
            "color:#a0a0a0;font-size:12px;background:transparent;")
        title_bar.addWidget(self.lbl_history_count)

        self.btn_clear_history = QPushButton("🗑 清空")
        self.btn_clear_history.setStyleSheet(
            "QPushButton{background:#795548;color:white;border:none;"
            "padding:6px 14px;font-size:12px;border-radius:5px;}"
            "QPushButton:hover{background:#5d4037;}")
        self.btn_clear_history.clicked.connect(self._clear_history)
        self.btn_clear_history.setFixedSize(80, 34)
        title_bar.addWidget(self.btn_clear_history)
        main_layout.addLayout(title_bar)

        # ---- 中间: 左侧历史列表 + 右侧详情 ----
        mid = QHBoxLayout()
        mid.setSpacing(6)

        # 左侧历史列表
        list_frame = QFrame()
        list_frame.setStyleSheet(
            "background:#16213e;border:1px solid #0f3460;border-radius:4px;")
        list_layout = QVBoxLayout(list_frame)
        list_layout.setContentsMargins(6, 6, 6, 6)

        self.history_list = QListWidget()
        self.history_list.setStyleSheet(
            "QListWidget{background:#16213e;color:#e0e0e0;"
            "border:none;font-size:12px;}"
            "QListWidget::item{padding:6px;border-bottom:1px solid #0f3460;}"
            "QListWidget::item:selected{background:#0f3460;}"
            "QListWidget::item:hover{background:#1a3a6e;}")
        self.history_list.currentRowChanged.connect(self._on_history_selected)
        self.history_list.setFixedWidth(260)
        self.history_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.history_list.setTextElideMode(Qt.ElideNone)
        list_layout.addWidget(self.history_list)
        mid.addWidget(list_frame)

        # 右侧详情区
        detail_frame = QFrame()
        detail_frame.setStyleSheet(
            "background:#16213e;border:1px solid #0f3460;border-radius:4px;")
        detail_layout = QVBoxLayout(detail_frame)
        detail_layout.setContentsMargins(8, 8, 8, 8)
        detail_layout.setSpacing(4)

        self.lbl_detail_title = QLabel("请选择一条记录查看详情")
        self.lbl_detail_title.setStyleSheet(
            "color:#e0e0e0;font-size:14px;font-weight:bold;background:transparent;")
        detail_layout.addWidget(self.lbl_detail_title)

        sep = QFrame(); sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color:#0f3460;"); detail_layout.addWidget(sep)

        # 滚动区域
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(
            "QScrollArea{background:transparent;border:none;}"
            "QScrollBar:vertical{background:#16213e;width:6px;}"
            "QScrollBar::handle:vertical{background:#0f3460;border-radius:3px;}")
        scroll_content = QWidget()
        scroll_content.setStyleSheet("background:transparent;")
        self._detail_content_layout = QVBoxLayout(scroll_content)
        self._detail_content_layout.setContentsMargins(0, 0, 0, 0)
        self._detail_content_layout.setSpacing(4)
        scroll.setWidget(scroll_content)
        detail_layout.addWidget(scroll, stretch=1)

        mid.addWidget(detail_frame, stretch=1)
        main_layout.addLayout(mid, stretch=1)

    def _mkbtn(self, text, bg, hover):
        btn = QPushButton(text)
        btn.setMinimumHeight(40)
        btn.setStyleSheet(
            f"QPushButton{{background:{bg};color:white;border:none;"
            f"padding:8px;font-size:14px;border-radius:6px;}}"
            f"QPushButton:hover{{background:{hover};}}"
            f"QPushButton:pressed{{background:{bg};}}")
        return btn

    # ============================================================
    # 检测线程管理
    # ============================================================
    def _start_detection(self):
        self._det_thread = DetectThread(self.args)
        self._det_thread.frame_ready.connect(self._on_frame)
        self._det_thread.status_update.connect(self._on_status)
        self._det_thread.start()

        if self._recording:
            QTimer.singleShot(1000, self._start_recording)

    def _start_recording(self):
        if self._current_frame is not None:
            h, w = self._current_frame.shape[:2]
            self.recorder.start_video(self.args.fps, (w, h))

    # ============================================================
    # 核心: 帧处理 - 自动截图 + 自动核验
    # ============================================================
    def _on_frame(self, frame, detections, fps):
        self._current_frame = frame
        self._current_dets = detections
        self._current_fps = fps

        # 过滤低置信度误检 (置信度阈值由 --conf 控制)
        detections = [d for d in detections if d[4] >= self.args.conf]

        # 过滤过小的误检框 (绝缘子目标通常较大)
        if self.args.min_area > 0:
            detections = [
                d for d in detections
                if (d[2] - d[0]) * (d[3] - d[1]) >= self.args.min_area
            ]

        # 分离包围大框: 大框只用于画面显示, 不参与核验/统计/结果
        small_dets, big_boxes = _classify_boxes(detections)

        # 判断是否有绝缘子、是否有缺陷 (只看小框/独立绝缘子)
        has_insulator = len(small_dets) > 0
        has_defect = any(int(d[5]) == 2 for d in small_dets)  # class 2(破损)才算缺陷
        now = time.time()

        # 触发 LLM 核验: 检测到绝缘子(任意类)就触发, 让LLM判断是否破损
        # (模型在近景下class 2识别不可靠, 借助LLM视觉能力弥补)
        cooldown_ok = (now - self._last_verify_time) > VERIFY_COOLDOWN
        will_verify = (has_insulator and self.verifier and self.verifier.enabled
                       and self._auto_verify_enabled and not self._verify_busy
                       and cooldown_ok)

        # --- 自动截图: 检测到绝缘子(任意类别)就截 ---
        if has_insulator and self._auto_snap_enabled:
            self._auto_snap_enabled = False
            snap_path = self.recorder.save_snapshot(frame)
            # 非核验截图只保存文件, 不加入历史记录, 避免结果页面数据过多
            if not will_verify:
                self._on_status(f"自动截图: {Path(snap_path).name}")
            # 1秒后可再次截图
            QTimer.singleShot(1000, self._reset_auto_snap)

        # --- 自动核验 (只用小框) ---
        if will_verify:
            self._verify_busy = True
            self._last_verify_time = now
            self.lbl_verify_status.setText("🔍 自动核验中...")
            self.lbl_verify_status.setStyleSheet(
                "color:#ff9800;font-size:12px;font-weight:bold;background:transparent;")

            # 后台线程只做API调用, 不碰任何Qt对象
            frame_copy = frame.copy()
            dets_copy = list(small_dets)

            # 兜底超时: 90秒后如果 _verify_busy 仍未重置, 强制释放
            _verify_id = getattr(self, '_verify_id', 0) + 1
            self._verify_id = _verify_id
            QTimer.singleShot(90000, lambda vid=_verify_id:
                self._verify_busy and vid == getattr(self, '_verify_id', 0)
                and self._on_verify_failed("超时(90s)"))

            def _auto_verify():
                try:
                    result = self.verifier.verify(frame_copy, dets_copy)
                    self._sig_verify_done.emit(result, dets_copy, frame_copy)
                except Exception as e:
                    print(f"[核验失败] API调用异常: {e}")
                    self._sig_verify_failed.emit(str(e))

            threading.Thread(target=_auto_verify, daemon=True).start()

        # 缩放帧到视频区域
        h, w = frame.shape[:2]
        r = min(VIDEO_W / w, VIDEO_H / h)
        nw, nh = int(w * r), int(h * r)
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        qimg = QImage(resized.data, nw, nh, nw * 3, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qimg))

        # 更新状态 (只看小框)
        self.lbl_fps.setText(f"FPS: {fps:.1f}")

        count_map = {}
        for d in small_dets:
            cls = int(d[5])
            count_map[cls] = count_map.get(cls, 0) + 1
        has_defect = any(c == 2 for c in count_map)  # 只有class 2(破损)算缺陷
        stats = " | ".join(f"{CLASS_CN.get(c,'?')}:{n}" for c, n in count_map.items())
        self.lbl_stats.setText("检测: " + (stats if stats else "无"))

        self.lbl_stats.setStyleSheet(
            "color:#ff4444;font-size:12px;font-weight:bold;background:transparent;"
            if has_defect else "color:#e0e0e0;font-size:12px;background:transparent;")

        # 更新 Web 仪表盘
        if self.web_server:
            self.web_server.update_state(
                frame, small_dets, fps, verify_busy=self._verify_busy,
                big_boxes=big_boxes)

        # 录像写入 (只记录小框)
        if self._recording and self.recorder:
            self.recorder.write_frame(frame, small_dets)

    def _reset_auto_snap(self):
        self._auto_snap_enabled = True

    # ============================================================
    # 核验完成回调 (运行在主线程)
    # ============================================================
    def _on_verify_safe(self, result, detections, frame):
        """从主线程安全调用, 完成UI更新和文件保存"""
        try:
            self._on_verify_done(result, detections, frame)
        except Exception as e:
            print(f"[核验失败] 处理异常: {e}")
            self._verify_busy = False
            self.lbl_verify_status.setText("✓ 自动核验就绪")
            self.lbl_verify_status.setStyleSheet(
                "color:#4caf50;font-size:12px;background:transparent;")
            self._on_status(f"核验处理异常: {e}")

    def _on_verify_failed(self, error_msg: str):
        """API调用失败时重置状态"""
        self._verify_busy = False
        self._on_status(f"核验失败: {error_msg}")
        self.lbl_verify_status.setText("✓ 自动核验就绪")
        self.lbl_verify_status.setStyleSheet(
            "color:#4caf50;font-size:12px;background:transparent;")

    def _on_verify_done(self, result, detections, frame):
        """核验完成: 先保存核验报告, 再添加到历史记录"""
        verify_time = time.time() - self._last_verify_time

        # 1. 保存截图 (如果之前没保存)
        snap_path = self.recorder.save_snapshot(frame)

        # 2. 保存核验报告 (核验完成后才保存)
        report_path = self.recorder.save_verify_report(
            detections, result, snapshot_path=snap_path)

        # 3. 解析核验结果
        verifications = result.get("verifications", [])
        correct = reject = uncertain = 0
        suggestion = ""
        for v in verifications:
            verdict = str(v.get("verification", "")).lower()
            if verdict == "correct":
                correct += 1
            elif verdict == "incorrect":
                reject += 1
            else:
                uncertain += 1
            if not suggestion and v.get("maintenance_suggestion"):
                suggestion = v.get("maintenance_suggestion")

        if not suggestion:
            for v in verifications:
                desc = v.get("defect_description", "")
                if desc and "建议" in desc:
                    suggestion = desc.split("建议")[-1]
                    break
        if not suggestion:
            suggestion = "请结合现场情况进一步确认,必要时安排巡检维护。"

        # 4. 构建历史记录条目
        record = {
            "id": len(self._history) + 1,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "target_count": len(detections),
            "correct_count": correct,
            "reject_count": reject,
            "uncertain_count": uncertain,
            "valid_count": len(verifications),
            "broken_count": sum(1 for d in detections if int(d[5]) == 2),
            "detections": [
                {
                    "class_id": int(d[5]),
                    "confidence": float(d[4]),
                    "bbox": [int(d[0]), int(d[1]), int(d[2]), int(d[3])],
                }
                for d in detections
            ],
            "verifications": verifications,
            "overall_assessment": result.get("overall_assessment", ""),
            "maintenance_suggestion": suggestion,
            "detect_time_ms": 0.0,
            "verify_time_ms": verify_time,
            "snapshot_path": snap_path,
            "report_path": report_path,
        }

        # 5. 添加到历史记录
        self._history.append(record)

        # 6. 构建文本报告
        lines = [
            "=" * 50,
            "  绝缘子缺陷检测 - LLM 智能核验报告",
            "=" * 50,
            f"时间: {record['timestamp']}",
            f"检测目标: {len(detections)} 个",
            f"核验引擎: {self.args.verify_api}",
            f"核验耗时: {verify_time:.2f}s",
            "-" * 50,
        ]
        for v in verifications:
            idx = v.get("target_index", "?")
            vrf = v.get("verification", "?")
            sev = v.get("severity", "none")
            desc = v.get("defect_description", "")
            sugg = v.get("maintenance_suggestion", "")
            lines.append(f"\n目标#{idx}:")
            lines.append(f"  核验结果: {vrf}")
            lines.append(f"  严重程度: {sev}")
            if desc:
                lines.append(f"  缺陷描述: {desc}")
            if sugg:
                lines.append(f"  维护建议: {sugg}")
        lines.append(f"\n{'='*50}")
        lines.append(f"整体评估: {result.get('overall_assessment', '')}")
        lines.append(f"维护建议: {suggestion}")
        lines.append(f"{'='*50}")
        lines.append(f"\n报告已保存: {report_path}")

        record["report_text"] = "\n".join(lines)

        # 7. 更新UI (当前已在主线程, 直接调用)
        verdict_msg = f"核验完成: 确认{correct}/误检{reject}/不确定{uncertain}"
        self._on_status(verdict_msg)
        self._on_verify_finished(all_reject=(reject > 0 and correct == 0))

    def _on_verify_finished(self, all_reject: bool = False):
        self._verify_busy = False
        if all_reject:
            self.lbl_verify_status.setText("✓ 核验就绪 (上轮:暂无缺陷)")
            self.lbl_verify_status.setStyleSheet(
                "color:#ff9800;font-size:12px;background:transparent;")
        else:
            self.lbl_verify_status.setText("✓ 自动核验就绪")
            self.lbl_verify_status.setStyleSheet(
                "color:#4caf50;font-size:12px;background:transparent;")

    def _add_snapshot_record(self, detections, snapshot_path):
        """把检测截图作为一条记录加入历史记录"""
        record = {
            "id": len(self._history) + 1,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "target_count": len(detections),
            "correct_count": 0,
            "reject_count": 0,
            "uncertain_count": 0,
            "valid_count": 0,
            "broken_count": sum(1 for d in detections if int(d[5]) == 2),
            "detections": [
                {
                    "class_id": int(d[5]),
                    "confidence": float(d[4]),
                    "bbox": [int(d[0]), int(d[1]), int(d[2]), int(d[3])],
                }
                for d in detections
            ],
            "verifications": [],
            "overall_assessment": "",
            "maintenance_suggestion": "",
            "detect_time_ms": 0.0,
            "verify_time_ms": 0.0,
            "snapshot_path": snapshot_path,
            "report_path": "",
        }
        self._history.append(record)

    # ============================================================
    # 控制按钮
    # ============================================================
    def _on_status(self, msg):
        self.lbl_status.setText(f"状态: {msg}")

    def _toggle_pause(self):
        if self._det_thread:
            paused = self._det_thread.toggle_pause()
            self.btn_pause.setText("▶ 开始" if paused else "⏸ 暂停")

    def _snapshot(self):
        if self._current_frame is None:
            self._on_status("无画面,无法截图")
            return
        path = self.recorder.save_snapshot(self._current_frame)
        if self._current_dets is not None:
            self._add_snapshot_record(self._current_dets, path)
        self._on_status(f"截图: {Path(path).name}")

    def _toggle_record(self):
        btn_s = ("QPushButton{{background:{bg};color:white;border:none;"
                 "padding:6px;font-size:12px;border-radius:5px;}}"
                 "QPushButton:hover{{background:{hover};}}")
        self._recording = not self._recording
        if self._recording:
            self.btn_record.setText("⏺ 录像中")
            self.btn_record.setStyleSheet(btn_s.format(bg="#ff5722", hover="#bf360c"))
            if self._current_frame is not None:
                h, w = self._current_frame.shape[:2]
                self.recorder.start_video(self.args.fps, (w, h))
            self._on_status("录像: 开始")
        else:
            self.btn_record.setText("⏺ 录像")
            self.btn_record.setStyleSheet(btn_s.format(bg="#607d8b", hover="#455a64"))
            self._on_status("录像: 停止")

    def _toggle_auto_verify(self):
        btn_s = ("QPushButton{{background:{bg};color:white;border:none;"
                 "padding:6px;font-size:12px;border-radius:5px;}}"
                 "QPushButton:hover{{background:{hover};}}")
        self._auto_verify_enabled = not self._auto_verify_enabled
        if self._auto_verify_enabled:
            self.btn_auto_toggle.setText("🔍 自动核验: ON")
            self.btn_auto_toggle.setStyleSheet(btn_s.format(bg="#4caf50", hover="#388e3c"))
            self.lbl_verify_status.setText("✓ 自动核验就绪")
            self.lbl_verify_status.setStyleSheet(
                "color:#4caf50;font-size:11px;background:transparent;")
        else:
            self.btn_auto_toggle.setText("🔍 自动核验: OFF")
            self.btn_auto_toggle.setStyleSheet(btn_s.format(bg="#607d8b", hover="#455a64"))
            self.lbl_verify_status.setText("⏸ 自动核验已关闭")
            self.lbl_verify_status.setStyleSheet(
                "color:#888;font-size:11px;background:transparent;")

    # ============================================================
    # 历史记录页面
    # ============================================================
    def _show_result_page(self):
        self._refresh_history_list()
        self.stack.setCurrentWidget(self.page_result)

    def _show_live_page(self):
        self.stack.setCurrentWidget(self.page_live)
        self.page_live.update()


    def _refresh_history_list(self):
        self.history_list.clear()
        for rec in reversed(self._history):
            rid = rec["id"]
            ts = rec["timestamp"]
            targets = rec["target_count"]
            correct = rec["correct_count"]
            reject = rec["reject_count"]
            broken = rec.get("broken_count",
                sum(1 for d in rec.get("detections",[]) if d.get("class_id")==2))
            total_v = rec["valid_count"]
            if broken > 0:
                icon = "⚠"
                item_text = f"{icon} #{rid} | {ts} | 检测:{targets} 破损:{broken} 核验:{total_v}"
            elif reject > 0:
                icon = "✗"
                item_text = f"{icon} #{rid} | {ts} | 检测:{targets} 误检:{reject} 核验:{total_v}"
            elif correct > 0:
                icon = "✓"
                item_text = f"{icon} #{rid} | {ts} | 检测:{targets} 正常 核验:{total_v}"
            else:
                icon = "?"
                item_text = f"{icon} #{rid} | {ts} | 检测:{targets} 未核验"
            item = QListWidgetItem(item_text)

            # 着色
            if broken > 0:
                item.setForeground(QColor(244, 67, 54))
            elif reject > 0:
                item.setForeground(QColor(255, 152, 0))
            elif correct > 0:
                item.setForeground(QColor(76, 175, 80))
            else:
                item.setForeground(QColor(160, 160, 160))
            self.history_list.addItem(item)

        self.lbl_history_count.setText(f"共 {len(self._history)} 条记录")

    def _on_history_selected(self, index):
        if index < 0 or index >= len(self._history):
            return

        # 反向索引 (列表显示是最新在上)
        rec_idx = len(self._history) - 1 - index
        rec = self._history[rec_idx]

        # 更新详情
        self.lbl_detail_title.setText(
            f"记录 #{rec['id']} - {rec['timestamp']}")

        # 清除旧详情
        while self._detail_content_layout.count():
            item = self._detail_content_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # ---- 现场截图 ----
        snap_path = rec.get("snapshot_path", "")
        if snap_path and Path(snap_path).is_file():
            try:
                img = cv2.imread(snap_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    h, w, _ = img.shape
                    # 缩放适配详情宽度 (~700px)
                    max_w = 680
                    if w > max_w:
                        ratio = max_w / w
                        img = cv2.resize(img, (max_w, int(h * ratio)))
                        h, w = img.shape[:2]
                    qimg = QImage(img.data, w, h, w * 3, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(qimg)
                    snap_label = QLabel()
                    snap_label.setPixmap(pixmap)
                    snap_label.setAlignment(Qt.AlignCenter)
                    snap_label.setStyleSheet(
                        "background:#000;border:1px solid #0f3460;border-radius:4px;")
                    self._detail_content_layout.addWidget(snap_label)
            except Exception:
                pass

        # 统计摘要
        broken = rec.get("broken_count",
            sum(1 for d in rec.get("detections",[]) if d.get("class_id")==2))
        summary = QLabel(
            f"检测目标: {rec['target_count']} | 破损: {broken} | "
            f"核验正确: {rec['correct_count']} | "
            f"误检: {rec['reject_count']} | "
            f"不确定: {rec['uncertain_count']} | "
            f"核验耗时: {rec['verify_time_ms']:.1f}s")
        summary.setStyleSheet("color:#e0e0e0;font-size:13px;background:transparent;")
        summary.setWordWrap(True)
        self._detail_content_layout.addWidget(summary)

        # 目标表格
        table = QTableWidget()
        table.setColumnCount(5)
        table.setHorizontalHeaderLabels(
            ["序号", "类型", "置信度", "核验结论", "严重程度"])
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        table.setStyleSheet(
            "QTableWidget{background:#1a1a2e;color:#e0e0e0;"
            "gridline-color:#0f3460;font-size:12px;border:1px solid #0f3460;}"
            "QHeaderView::section{background:#0f3460;color:#fff;padding:4px;"
            "font-weight:bold;border:1px solid #1a1a2e;}"
            "QTableWidget::item{padding:4px;border:1px solid #0f3460;}")
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.verticalHeader().setVisible(False)

        dets = rec.get("detections", [])
        veris = rec.get("verifications", [])
        veri_map = {v.get("target_index", i+1): v for i, v in enumerate(veris)}
        table.setRowCount(len(dets))
        for i, det in enumerate(dets):
            cls = int(det.get("class_id", 0))
            conf = float(det.get("confidence", 0))
            v = veri_map.get(i+1, {})
            verdict = v.get("verification", "未核验")
            severity = v.get("severity", "none")

            table.setItem(i, 0, QTableWidgetItem(str(i+1)))
            table.setItem(i, 1, QTableWidgetItem(CLASS_CN.get(cls, "?")))
            table.setItem(i, 2, QTableWidgetItem(f"{conf:.2f}"))
            table.setItem(i, 3, QTableWidgetItem(str(verdict)))
            table.setItem(i, 4, QTableWidgetItem(str(severity)))

            color = QColor(224, 224, 224)
            if verdict == "correct":
                color = QColor(76, 175, 80)
            elif verdict == "incorrect":
                color = QColor(244, 67, 54)
            elif verdict == "uncertain":
                color = QColor(255, 152, 0)
            for c in range(5):
                table.item(i, c).setForeground(color)

        table.setMaximumHeight(max(80, len(dets) * 36 + 30))
        self._detail_content_layout.addWidget(table)

        # 综合评价 (已包含维护建议)
        overall_label = QLabel("📋 综合评估:")
        overall_label.setStyleSheet(
            "color:#00ff88;font-size:14px;font-weight:bold;background:transparent;margin-top:6px;")
        self._detail_content_layout.addWidget(overall_label)

        overall_text = QLabel(rec.get("overall_assessment", "暂无") or "暂无")
        overall_text.setStyleSheet(
            "color:#e0e0e0;font-size:12px;background:transparent;line-height:1.6;")
        overall_text.setWordWrap(True)
        overall_text.setFixedWidth(420)
        self._detail_content_layout.addWidget(overall_text)

        # 文件路径
        file_info = QLabel(
            f"截图: {rec.get('snapshot_path', 'N/A')}\n"
            f"报告: {rec.get('report_path', 'N/A')}")
        file_info.setStyleSheet(
            "color:#888;font-size:10px;background:transparent;margin-top:4px;")
        file_info.setWordWrap(True)
        self._detail_content_layout.addWidget(file_info)

        self._detail_content_layout.addStretch()

    def _clear_history(self):
        self._history.clear()
        self._refresh_history_list()
        self.lbl_detail_title.setText("请选择一条记录查看详情")
        while self._detail_content_layout.count():
            item = self._detail_content_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    # ============================================================
    # 其他
    # ============================================================
    def _update_time(self):
        self.lbl_time.setText(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def _exit(self):
        if self._det_thread:
            self._det_thread.stop()
        if self.recorder:
            self.recorder.release()
        self.close()

    def closeEvent(self, event):
        if self._det_thread:
            self._det_thread.stop()
        if self.recorder:
            self.recorder.release()
        event.accept()


# ============================================================
# 命令行
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="ELF2 绝缘子缺陷检测系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python3 main.py                                                    # 默认启动
  python3 main.py --camera /dev/video11 --model model.rknn           # 指定参数
  python3 main.py --conf 0.15 --record                               # 调阈值+录像
  python3 main.py --verify-api zhipu                                 # 启用 LLM 自动核验
        """)
    p.add_argument("--camera", default="/dev/video11")
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--model", default="model.rknn")
    p.add_argument("--conf", type=float, default=0.25,
                   help="置信度阈值 (通用, 越高误检越少)")
    p.add_argument("--defect-conf", type=float, default=0.25,
                   help="缺陷类置信度阈值 (class 1~4专用, 默认0.25, 低于此值不标红框)")
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--input-size", type=int, default=640)
    p.add_argument("--min-area", type=int, default=8000,
                   help="最小检测框面积(像素), 过滤过小误检, 默认8000, 0表示关闭")
    p.add_argument("--record", action="store_true", help="启动录像")
    p.add_argument("--output-dir", default="records")
    p.add_argument("--verify-api", choices=["zhipu", ""], default="",
                   help="LLM 核验 API: zhipu (需设置 ZHIPU_API_KEY 环境变量)")
    p.add_argument("--web", action="store_true",
                   help="启动 Web 仪表盘 (PC 浏览器访问 http://<板端IP>:5000)")
    p.add_argument("--web-port", type=int, default=5000,
                   help="Web 仪表盘端口, 默认 5000")
    p.add_argument("--test-dir", default="",
                   help="测试模式: 从文件夹随机抽取5张图片进行检测核验, 不走摄像头")
    return p.parse_args()


# ============================================================
# 入口
# ============================================================
def main():
    args = parse_args()

    # 切换到脚本目录
    os.chdir(Path(__file__).parent)

    # 自动设置 Wayland 环境
    if not os.environ.get("QT_QPA_PLATFORM"):
        if os.path.exists("/run/user/1000/wayland-0"):
            os.environ["QT_QPA_PLATFORM"] = "wayland"
            os.environ["WAYLAND_DISPLAY"] = "wayland-0"
            os.environ["XDG_RUNTIME_DIR"] = "/run/user/1000"

    # 检查 PyQt5
    if not _init_qt():
        print("[ERROR] PyQt5 未安装, 请运行: sudo apt install python3-pyqt5")
        sys.exit(1)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # 暗色主题
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(26, 26, 46))
    palette.setColor(QPalette.WindowText, QColor(224, 224, 224))
    palette.setColor(QPalette.Base, QColor(22, 33, 62))
    palette.setColor(QPalette.Text, QColor(224, 224, 224))
    palette.setColor(QPalette.Button, QColor(22, 33, 62))
    palette.setColor(QPalette.ButtonText, QColor(224, 224, 224))
    palette.setColor(QPalette.Highlight, QColor(233, 69, 96))
    app.setPalette(palette)

    window = MainWindow(args)
    window.show()

    # 信号处理
    def sig_handler(sig, frame):
        window._exit()
        sys.exit(0)
    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

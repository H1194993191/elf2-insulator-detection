#!/usr/bin/env python3
"""
绝缘子检测批量测试 + 实验数据汇总
==================================
功能:
  1. 遍历指定目录所有图片, 逐张 NPU 推理 + LLM 核验
  2. 每张结果保存为独立 JSON (reports/ 目录)
  3. 汇总结果保存为 summary.json
  4. 自动生成 6 张实验图表 + 1 张汇总表 (输出到 --charts-dir)

用法:
  # 板端运行 (NPU + LLM)
  python3 test_all_images.py \
    --image-dir /path/to/images \
    --model model.rknn \
    --verify-api zhipu

  # PC端运行 (仅 LLM 核验, 无 NPU)
  python3 test_all_images.py \
    --image-dir C:/Users/Hqb/Desktop/ELF_测试图片 \
    --verify-api zhipu \
    --no-npu

  # 仅生成图表 (已有 summary.json)
  python3 test_all_images.py \
    --summary summary.json \
    --charts-only
"""

import argparse
import base64
import glob
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import cv2
import numpy as np

# ── 可选依赖 ──
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    from rknnlite.api import RKNNLite
    HAS_RKNN = True
except ImportError:
    HAS_RKNN = False


# ============================================================
# 常量
# ============================================================
CLASS_NAMES = {0: "Normal", 1: "JYZ", 2: "Broken"}
CLASS_CN    = {0: "正常", 1: "绝缘子", 2: "破损"}
CLASS_COLORS = {0: "#4caf50", 1: "#00bcd4", 2: "#f44336"}
CLASS_COLORS_BGR = {0: (0, 255, 0), 1: (255, 255, 0), 2: (0, 0, 255)}

# ============================================================
# NPU 检测器 (精简版, 从 main.py 提取)
# ============================================================
def _letterbox(img, new_shape=640):
    shape = img.shape[:2]
    r = min(new_shape / shape[0], new_shape / shape[1])
    nu = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = (new_shape - nu[0]) / 2, (new_shape - nu[1]) / 2
    if shape[::-1] != nu:
        img = cv2.resize(img, nu, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    return cv2.copyMakeBorder(img, top, bottom, left, right,
                               cv2.BORDER_CONSTANT, value=(114, 114, 114)), r, (dw, dh)


def _nms(boxes, scores, iou_thres=0.45):
    if len(boxes) == 0:
        return []
    x1, y1 = boxes[:, 0], boxes[:, 1]
    x2, y2 = boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    idxs = np.argsort(scores)[::-1]
    keep = []
    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[idxs[1:]])
        yy1 = np.maximum(y1[i], y1[idxs[1:]])
        xx2 = np.minimum(x2[i], x2[idxs[1:]])
        yy2 = np.minimum(y2[i], y2[idxs[1:]])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        iou = (w * h) / (areas[i] + areas[idxs[1:]] - w * h + 1e-6)
        idxs = idxs[1:][iou <= iou_thres]
    return keep


class NPUDetector:
    """RKNN NPU 推理封装"""

    def __init__(self, model_path: str, conf_thres=0.25, iou_thres=0.45,
                 input_size=640):
        self.model_path = model_path
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.input_size = input_size
        self._rknn = None

    def load(self) -> bool:
        if not HAS_RKNN:
            print("[NPU] rknn-toolkit-lite2 未安装, 跳过 NPU 推理")
            return False
        try:
            self._rknn = RKNNLite()
            ret = self._rknn.load_rknn(self.model_path)
            if ret != 0:
                print(f"[NPU] 加载模型失败: {ret}")
                return False
            ret = self._rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
            if ret != 0:
                print(f"[NPU] 初始化 runtime 失败: {ret}")
                return False
            print("[NPU] 模型加载成功")
            return True
        except Exception as e:
            print(f"[NPU] 加载异常: {e}")
            return False

    def detect(self, frame: np.ndarray) -> List[List]:
        if self._rknn is None:
            return []
        oh, ow = frame.shape[:2]
        img, r, pad = _letterbox(frame, self.input_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_in = np.expand_dims(img, axis=0)
        try:
            outputs = self._rknn.inference(inputs=[img_in])
        except Exception as e:
            print(f"[NPU] 推理异常: {e}")
            return []
        return self._postprocess(outputs, ow, oh, r, pad)

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
        bb, cid, cc = bb[mask], cid[mask], cc[mask]
        if np.max(bb) < 2.0:
            bb = bb * self.input_size
        x, y, w, h = bb[:, 0], bb[:, 1], bb[:, 2], bb[:, 3]
        dw, dh = pad
        x1 = np.clip((x - w / 2 - dw) / r, 0, ow - 1)
        y1 = np.clip((y - h / 2 - dh) / r, 0, oh - 1)
        x2 = np.clip((x + w / 2 - dw) / r, 0, ow - 1)
        y2 = np.clip((y + h / 2 - dh) / r, 0, oh - 1)
        boxes = np.stack([x1, y1, x2, y2], axis=1)
        keep = _nms(boxes, cc, self.iou_thres)
        return [[float(boxes[i, 0]), float(boxes[i, 1]), float(boxes[i, 2]),
                 float(boxes[i, 3]), float(cc[i]), int(cid[i])] for i in keep]

    def release(self):
        if self._rknn:
            self._rknn.release()
            self._rknn = None


# ============================================================
# LLM 核验 (智谱 GLM-4V)
# ============================================================
class LLMVerifier:
    def __init__(self):
        self.api_key = os.environ.get("ZHIPU_API_KEY", "")
        self._enabled = bool(self.api_key)

    def verify(self, frame: np.ndarray, detections: List) -> Dict:
        if not self._enabled or not detections:
            return {"verifications": [], "overall_assessment": ""}

        # 画框 + 编码
        vis = frame.copy()
        for d in detections:
            x1, y1, x2, y2 = int(d[0]), int(d[1]), int(d[2]), int(d[3])
            cls = int(d[5])
            color = CLASS_COLORS_BGR.get(cls, (0, 255, 0))
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            label = f"{CLASS_NAMES.get(cls, '?')} {d[4]:.2f}"
            cv2.putText(vis, label, (x1, max(y1 - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        # 缩小图片
        h, w = vis.shape[:2]
        if max(h, w) > 1024:
            scale = 1024 / max(h, w)
            vis = cv2.resize(vis, (int(w * scale), int(h * scale)))
        _, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
        img_b64 = base64.b64encode(buf).decode("utf-8")

        det_lines = []
        for i, d in enumerate(detections, 1):
            cls = int(d[5])
            x1, y1, x2, y2 = int(d[0]), int(d[1]), int(d[2]), int(d[3])
            det_lines.append(
                f"  目标#{i}: class={CLASS_NAMES.get(cls,'?')}({CLASS_CN.get(cls,'?')}), "
                f"conf={d[4]:.3f}, bbox=({x1},{y1},{x2},{y2})")

        prompt = (
            "你是电力巡检视觉专家。请逐项核验图片中用色框标注的绝缘子目标是否存在破损/缺陷。\n\n"
            "检测目标 (绿框=正常, 蓝框=绝缘子, 红框=破损):\n"
            + "\n".join(det_lines) +
            "\n\n严格以下JSON格式返回(只返回JSON):\n"
            "{\n"
            '  "verifications": [\n'
            '    {"target_index":1,"verification":"correct|incorrect|uncertain",'
            '"severity":"none|minor|moderate|severe|critical",'
            '"defect_description":"...","maintenance_suggestion":"..."},\n'
            '    ...\n'
            '  ],\n'
            '  "overall_assessment": "整体评估总结"\n'
            "}"
        )

        import requests
        try:
            resp = requests.post(
                "https://open.bigmodel.cn/api/paas/v4/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "glm-4v",
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url",
                             "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                        ]
                    }],
                    "temperature": 0.1,
                    "max_tokens": 2000
                },
                timeout=90
            )
            if resp.status_code != 200:
                print(f"[LLM] API 返回 {resp.status_code}: {resp.text[:200]}")
                return {"verifications": [], "overall_assessment": f"HTTP {resp.status_code}"}
            content = resp.json()["choices"][0]["message"]["content"]
            content = content.strip()
            # 提取 JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            result = json.loads(content)
            return result
        except Exception as e:
            print(f"[LLM] 核验失败: {e}")
            return {"verifications": [], "overall_assessment": str(e)}


# ============================================================
# 图表生成
# ============================================================
def generate_charts(summary: Dict, output_dir: str):
    """生成 6 张实验图表 + 1 个汇总 CSV"""
    if not HAS_MPL:
        print("[警告] matplotlib 未安装, 跳过图表生成")
        return

    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'WenQuanYi Micro Hei']
    plt.rcParams['axes.unicode_minus'] = False

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── 汇总表 (CSV) ──
    _save_summary_csv(summary, out)

    # ── 图1: 置信度分布直方图 ──
    _chart_confidence_distribution(summary, out)

    # ── 图2: 各类别检测数量柱状图 ──
    _chart_class_counts(summary, out)

    # ── 图3: LLM 核验结果饼图 ──
    _chart_verification_pie(summary, out)

    # ── 图4: 缺陷严重度分布 ──
    _chart_severity_bar(summary, out)

    # ── 图5: 每图检测数分布 ──
    _chart_detections_per_image(summary, out)

    # ── 图6: 推理耗时统计 ──
    _chart_timing(summary, out)

    print(f"图表已保存至: {out}")


def _save_summary_csv(summary: Dict, out_dir: Path):
    """保存 CSV 汇总表"""
    path = out_dir / "summary_table.csv"
    hdr = ["指标", "值"]
    rows = [
        hdr,
        ["测试图片总数", str(summary.get("total_images", 0))],
        ["成功处理图片数", str(summary.get("processed_images", 0))],
        ["总检测框数", str(summary.get("total_detections", 0))],
        ["平均每图检测数", f"{summary.get('avg_detections_per_image', 0):.2f}"],
        ["Normal (正常) 检测数", str(summary.get("class_counts", {}).get("Normal", 0))],
        ["JYZ (绝缘子) 检测数", str(summary.get("class_counts", {}).get("JYZ", 0))],
        ["Broken (破损) 检测数", str(summary.get("class_counts", {}).get("Broken", 0))],
        ["LLM 核验次数", str(summary.get("total_verifications", 0))],
        ["确认正确 (correct)", str(summary.get("verification_stats", {}).get("correct", 0))],
        ["误检 (incorrect)", str(summary.get("verification_stats", {}).get("incorrect", 0))],
        ["不确定 (uncertain)", str(summary.get("verification_stats", {}).get("uncertain", 0))],
        ["平均 NPU 推理耗时 (ms)", f"{summary.get('avg_detect_time_ms', 0):.1f}"],
        ["平均 LLM 核验耗时 (s)", f"{summary.get('avg_verify_time_s', 0):.2f}"],
        ["总耗时 (s)", f"{summary.get('total_time_s', 0):.1f}"],
    ]
    with open(path, "w", encoding="utf-8-sig") as f:
        for row in rows:
            f.write(",".join(row) + "\n")
    print(f"  [OK] {path.name}")


def _chart_confidence_distribution(summary: Dict, out_dir: Path):
    per_class = summary.get("confidence_by_class", {})
    if not per_class:
        return
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for idx, (cls_name, confs) in enumerate([("Normal", []), ("JYZ", []), ("Broken", [])]):
        ax = axes[idx]
        data = per_class.get(cls_name, [])
        color = CLASS_COLORS.get(idx, "#888888")
        if data:
            ax.hist(data, bins=20, color=color, alpha=0.8, edgecolor='white')
            ax.axvline(np.mean(data), color='red', linestyle='--', linewidth=1.5,
                       label=f'Mean {np.mean(data):.3f}')
            ax.legend(fontsize=8)
        ax.set_title(f"{cls_name}  n={len(data)}", fontsize=10)
        ax.set_xlabel("Confidence", fontsize=9)
        ax.set_ylabel("Frequency", fontsize=9)
        ax.set_xlim(0, 1)
    fig.suptitle("Confidence Distribution by Class", fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_dir / "01_confidence_distribution.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] 01_confidence_distribution.png")


def _chart_class_counts(summary: Dict, out_dir: Path):
    counts = summary.get("class_counts", {})
    if not counts:
        return
    names = list(counts.keys())
    values = [counts[n] for n in names]
    colors = [CLASS_COLORS.get({"Normal": 0, "JYZ": 1, "Broken": 2}.get(n, 0), "#888") for n in names]
    cn_labels = [n for n in names]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(cn_labels, values, color=colors, alpha=0.85, edgecolor='white', width=0.5)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.02,
                str(v), ha='center', fontsize=12, fontweight='bold')
    ax.set_ylabel("Detection Count", fontsize=11)
    ax.set_title("Detection Count by Class", fontsize=13, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "02_class_counts.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] 02_class_counts.png")


def _chart_verification_pie(summary: Dict, out_dir: Path):
    stats = summary.get("verification_stats", {})
    if not stats:
        return
    labels_all = ["Correct", "Incorrect", "Uncertain"]
    values_all = [stats.get("correct", 0), stats.get("incorrect", 0), stats.get("uncertain", 0)]
    if sum(values_all) == 0:
        print("  [SKIP] LLM verification pie (no data)")
        return
    colors_all = ["#4caf50", "#f44336", "#ff9800"]

    # 过滤值为 0 的类别，避免空切片导致文字重叠
    filtered = [(l, v, c) for l, v, c in zip(labels_all, values_all, colors_all) if v > 0]
    labels = [x[0] for x in filtered]
    values = [x[1] for x in filtered]
    colors = [x[2] for x in filtered]
    explode = tuple(0.03 for _ in values)

    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(
        values, labels=labels, colors=colors,
        autopct='%1.1f%%', startangle=90, pctdistance=0.6,
        textprops={'fontsize': 12})
    for at in autotexts:
        at.set_fontweight('bold')
        at.set_fontsize(14)
    ax.set_title(f"LLM Verification Results (total: {sum(values)})", fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_dir / "03_verification_pie.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] 03_verification_pie.png")


def _chart_severity_bar(summary: Dict, out_dir: Path):
    sevs = summary.get("severity_counts", {})
    if not sevs:
        return
    order = ["none", "minor", "moderate", "severe", "critical"]
    cn_order = ["None", "Minor", "Moderate", "Severe", "Critical"]
    colors = ["#4caf50", "#8bc34a", "#ff9800", "#f44336", "#b71c1c"]
    values = [sevs.get(k, 0) for k in order]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(cn_order, values, color=colors, alpha=0.85, edgecolor='white', width=0.55)
    for bar, v in zip(bars, values):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.03,
                    str(v), ha='center', fontsize=11, fontweight='bold')
    ax.set_ylabel("Object Count", fontsize=11)
    ax.set_title("Defect Severity Distribution", fontsize=13, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "04_severity_distribution.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] 04_severity_distribution.png")


def _chart_detections_per_image(summary: Dict, out_dir: Path):
    per_img = summary.get("detections_per_image", [])
    if not per_img:
        return

    # 前30张 (太多了看不清)
    per_img = per_img[:30]
    img_names = [f"#{i+1}" for i in range(len(per_img))]

    normal_vals = [d.get("Normal", 0) for d in per_img]
    jyz_vals = [d.get("JYZ", 0) for d in per_img]
    broken_vals = [d.get("Broken", 0) for d in per_img]

    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(per_img))
    width = 0.6
    p1 = ax.bar(x, normal_vals, width, color="#4caf50", alpha=0.85, label="Normal")
    p2 = ax.bar(x, jyz_vals, width, bottom=normal_vals, color="#00bcd4", alpha=0.85, label="JYZ")
    bottom2 = [n + j for n, j in zip(normal_vals, jyz_vals)]
    p3 = ax.bar(x, broken_vals, width, bottom=bottom2, color="#f44336", alpha=0.85, label="Broken")

    ax.set_xticks(x[::2])
    ax.set_xticklabels(img_names[::2], fontsize=8, rotation=45)
    ax.set_ylabel("Detection Count", fontsize=11)
    ax.set_title("Detections per Image (top 30, stacked by class)", fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "05_detections_per_image.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] 05_detections_per_image.png")


def _chart_timing(summary: Dict, out_dir: Path):
    detect_times = summary.get("detect_times_ms", [])
    verify_times = summary.get("verify_times_s", [])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # NPU inference time
    ax = axes[0]
    if detect_times:
        ax.hist(detect_times, bins=20, color="#2196F3", alpha=0.8, edgecolor='white')
        ax.axvline(np.mean(detect_times), color='red', linestyle='--', linewidth=1.5,
                   label=f'Mean {np.mean(detect_times):.1f} ms')
        ax.legend(fontsize=9)
    ax.set_title("NPU Inference Time", fontsize=11)
    ax.set_xlabel("Time (ms)", fontsize=9)
    ax.set_ylabel("Frequency", fontsize=9)

    # LLM verification time
    ax = axes[1]
    if verify_times:
        ax.hist(verify_times, bins=15, color="#FF9800", alpha=0.8, edgecolor='white')
        ax.axvline(np.mean(verify_times), color='red', linestyle='--', linewidth=1.5,
                   label=f'Mean {np.mean(verify_times):.1f} s')
        ax.legend(fontsize=9)
    ax.set_title("LLM Verification Time", fontsize=11)
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel("Frequency", fontsize=9)

    fig.suptitle("Inference Time Analysis", fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_dir / "06_timing_stats.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] 06_timing_stats.png")


# ============================================================
# 主流程
# ============================================================
def run_test(args):
    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir) if args.output_dir else image_dir / "_test_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir = output_dir / "reports"
    report_dir.mkdir(exist_ok=True)

    # ── 扫描图片 ──
    exts = ["**/*.jpg", "**/*.jpeg", "**/*.png", "**/*.bmp"]
    images = []
    for ext in exts:
        images.extend(image_dir.glob(ext))
        images.extend(image_dir.glob(ext.upper()))
    # 排除输出目录内的图片（如之前生成的图表）
    images = [p for p in images if "_test_results" not in str(p)]
    images = sorted(images)

    if not images:
        print(f"[错误] 目录无图片: {image_dir}")
        return

    print(f"[信息] 找到 {len(images)} 张图片")
    print(f"[信息] 输出目录: {output_dir}")

    # ── 初始化 ──
    detector = None
    if not args.no_npu:
        detector = NPUDetector(
            model_path=args.model, conf_thres=args.conf,
            iou_thres=args.iou, input_size=args.input_size)
        if not detector.load():
            print("[警告] NPU 未就绪, 将跳过目标检测")
            detector = None

    verifier = LLMVerifier()
    if not verifier._enabled:
        print("[警告] ZHIPU_API_KEY 未设置, 将跳过 LLM 核验")
        print("       请设置: export ZHIPU_API_KEY=\"xxx.xxx\"")

    # ── 收集器 ──
    all_results = []
    all_confidences = {"Normal": [], "JYZ": [], "Broken": []}
    all_severity = {"none": 0, "minor": 0, "moderate": 0, "severe": 0, "critical": 0}
    verify_stats = {"correct": 0, "incorrect": 0, "uncertain": 0}
    detect_times = []
    verify_times_list = []
    detections_per_image = []
    class_total = {"Normal": 0, "JYZ": 0, "Broken": 0}

    t_start = time.time()

    for idx, img_path in enumerate(images, 1):
        print(f"\n[{idx}/{len(images)}] {img_path.name}", end=" ", flush=True)
        frame = cv2.imread(str(img_path))
        if frame is None:
            print("→ 无法读取, 跳过")
            continue

        t0 = time.time()
        detections = detector.detect(frame) if detector else []
        dt = (time.time() - t0) * 1000
        detect_times.append(dt)

        print(f"→ NPU检测 {len(detections)} 个目标 ({dt:.1f}ms)", end=" ", flush=True)

        # 收集置信度
        img_class_counts = {"Normal": 0, "JYZ": 0, "Broken": 0}
        for d in detections:
            cls = CLASS_NAMES.get(int(d[5]), "?")
            if cls in all_confidences:
                all_confidences[cls].append(float(d[4]))
            if cls in class_total:
                class_total[cls] += 1
            img_class_counts[cls] = img_class_counts.get(cls, 0) + 1
        detections_per_image.append(img_class_counts)

        # LLM 核验
        result = {}
        vt = 0
        if verifier._enabled and detections:
            tv0 = time.time()
            result = verifier.verify(frame, detections)
            vt = time.time() - tv0
            verify_times_list.append(vt)
            print(f"→ LLM核验 {vt:.1f}s", end=" ")

            # 统计核验结果
            for v in result.get("verifications", []):
                verdict = str(v.get("verification", "")).lower()
                if verdict in verify_stats:
                    verify_stats[verdict] += 1
                sev = str(v.get("severity", "none")).lower()
                if sev in all_severity:
                    all_severity[sev] += 1
        else:
            print("→ (无LLM)", end=" ")

        # 保存单张结果
        record = {
            "image": img_path.name,
            "image_path": str(img_path),
            "detections": [{
                "class": CLASS_NAMES.get(int(d[5]), "?"),
                "class_cn": CLASS_CN.get(int(d[5]), "?"),
                "confidence": float(d[4]),
                "bbox": [int(d[0]), int(d[1]), int(d[2]), int(d[3])]
            } for d in detections],
            "verifications": result.get("verifications", []),
            "overall_assessment": result.get("overall_assessment", ""),
            "detect_time_ms": dt,
            "verify_time_s": vt,
        }
        report_path = report_dir / f"{img_path.stem}_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)

        all_results.append(record)
        print(f"→ 已保存")

        # LLM API 请求间隔，避免触发频率限制
        if verifier._enabled and detections:
            time.sleep(2)

    t_total = time.time() - t_start

    # ── 汇总 ──
    summary = {
        "test_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "test_dir": str(image_dir),
        "total_images": len(images),
        "processed_images": len(all_results),
        "total_detections": sum(class_total.values()),
        "avg_detections_per_image": sum(class_total.values()) / max(1, len(all_results)),
        "class_counts": class_total,
        "confidence_by_class": {k: v for k, v in all_confidences.items() if v},
        "total_verifications": sum(verify_stats.values()),
        "verification_stats": verify_stats,
        "severity_counts": all_severity,
        "avg_detect_time_ms": np.mean(detect_times) if detect_times else 0,
        "avg_verify_time_s": np.mean(verify_times_list) if verify_times_list else 0,
        "total_time_s": t_total,
        "detect_times_ms": detect_times,
        "verify_times_s": verify_times_list,
        "detections_per_image": detections_per_image,
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[汇总] → {summary_path}")
    print(f"  总图片: {summary['total_images']}")
    print(f"  总检测: {summary['total_detections']}  (Normal:{class_total['Normal']} JYZ:{class_total['JYZ']} Broken:{class_total['Broken']})")
    print(f"  总耗时: {t_total:.1f}s")
    if verify_stats:
        print(f"  核验: 正确{verify_stats['correct']} 误检{verify_stats['incorrect']} 不确定{verify_stats['uncertain']}")

    # ── 生成图表 ──
    charts_dir = args.charts_dir or str(output_dir / "charts")
    generate_charts(summary, charts_dir)

    if detector:
        detector.release()

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="绝缘子检测批量测试 + 实验数据汇总")
    parser.add_argument("--image-dir", default=r"C:\Users\Hqb\Desktop\ELF_测试图片",
                        help="测试图片目录")
    parser.add_argument("--model", default="model.rknn", help="RKNN 模型路径")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU 阈值")
    parser.add_argument("--input-size", type=int, default=640, help="模型输入尺寸")
    parser.add_argument("--verify-api", choices=["zhipu"], help="启用 LLM 核验")
    parser.add_argument("--no-npu", action="store_true", help="不使用 NPU (仅 LLM 核验)")
    parser.add_argument("--output-dir", default="", help="结果输出目录 (默认: 图片目录/_test_results)")
    parser.add_argument("--charts-dir", default="", help="图表输出目录 (默认: 输出目录/charts)")
    parser.add_argument("--charts-only", action="store_true", help="仅从 summary.json 生成图表")
    parser.add_argument("--summary", default="", help="已有的 summary.json 路径 (配合 --charts-only)")

    args = parser.parse_args()

    if args.charts_only:
        if not args.summary:
            print("[错误] --charts-only 需要 --summary 参数")
            sys.exit(1)
        summary_path = Path(args.summary)
        if not summary_path.exists():
            print(f"[错误] 文件不存在: {args.summary}")
            sys.exit(1)
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        out_dir = args.charts_dir or str(summary_path.parent / "charts")
        generate_charts(summary, out_dir)
    else:
        run_test(args)

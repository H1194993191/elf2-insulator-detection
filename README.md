# ELF2 绝缘子缺陷检测系统

> 基于 **瑞芯微 RK3588** 的边缘计算绝缘子缺陷检测与智能核验系统

---

## 项目概览

```
OV13855 摄像头 → RKNN NPU 推理 (3分类) → Qt 触屏实时显示
                                      → 自动截图 (1秒冷却)
                                      → 自动调用 LLM 核验 (2秒冷却) → 自动保存报告
                                      → Web 仪表盘 (MJPEG 推流 + 历史查看)
```

### 3 类检测目标

| ID | 英文名 | 中文名 | 框颜色 | 说明 |
|:--:|--------|--------|--------|------|
| 0 | Nora1 | 正常 | 🟢 绿色 | 绝缘子状态正常，无破损 |
| 1 | JYZ | 绝缘子 | 🔵 蓝色 | 绝缘子检测目标（非缺陷类） |
| 2 | Broken | 破损 | 🔴 红色 | 绝缘子本体破损 / 裂纹 / 缺陷 |

> **注意：** 只有 class 2 (Broken) 算作缺陷，class 0/1 均为非缺陷类。
> 模型在近景下 class 2 识别精度不够稳定，因此所有检测目标均会触发 LLM 二次核验，利用大模型视觉能力弥补。

---

## 硬件平台

| 组件 | 型号 |
|------|------|
| 主板 | ELF 2 (RK3588, 8GB+64GB) |
| NPU | RK3588 NPU (3核, 6 TOPS) |
| 摄像头 | OV13855 MIPI-CSI (`/dev/video11`, rkisp_mainpath) |
| 触摸屏 | 7 英寸 MIPI LCD (1024×600, Wayland) |
| 无线 | CF-AX200-M (WiFi / 蓝牙) |

---

## 快速开始

### 环境要求

- Ubuntu 20.04+ (板端 ARM64)
- Python 3.10
- `rknn-toolkit-lite2` (板端 NPU 推理)
- PyQt5, OpenCV, NumPy, Flask

### 安装依赖

```bash
cd /home/elf/RK3588/project
pip install -r requirements.txt

# RK3588 板端额外安装 rknn-toolkit-lite2 (从瑞芯微官方获取 .whl)
# pip install rknn_toolkit_lite2-*-cp310-cp310-linux_aarch64.whl
```

### 运行

```bash
# 默认启动 (摄像头 + Qt 触屏界面)
python3 main.py

# 调低检测阈值 + 开启录像
python3 main.py --conf 0.15 --record

# 启用 LLM 核验 + Web 仪表盘
export ZHIPU_API_KEY="xxx.xxx"
python3 main.py --verify-api zhipu --web

# 使用 DeepSeek 核验
export DEEPSEEK_API_KEY="sk-xxx"
python3 main.py --verify-api deepseek --web

# 测试模式: 从文件夹随机抽取 5 张图片进行检测和核验 (不走摄像头)
python3 main.py --test-dir /path/to/images --verify-api zhipu
```

Web 仪表盘启动后，PC 浏览器访问：
- 实时监控：`http://<板端IP>:5000`
- 核验结果：`http://<板端IP>:5000/results`

---

## 功能说明

### 1. Qt 触屏界面

**实时检测页：**

| 按钮 | 功能 |
|------|------|
| ⏸ 暂停 / ▶ 开始 | 暂停 / 恢复检测 |
| 📷 截图 | 手动保存当前画面到 `records/snapshots/` |
| ⏺ 录像 | 开始 / 停止录像 (`records/video_*.mp4`) |
| 🔍 自动核验: ON/OFF | 切换是否自动触发 LLM 核验 |
| 📊 历史记录 | 切换到历史记录页面 |
| ✕ 退出 | 安全退出系统 |

状态栏实时显示：FPS、各类检测数量、核验状态。

**自动流程（无需手动操作）：**

1. 检测到任意绝缘子 → **自动截图**（1 秒冷却）
2. 检测到绝缘子 → **自动调用 LLM 核验**（2 秒冷却，借助 LLM 视觉能力判断是否破损）
3. 核验完成 → **自动保存 JSON 报告** + 添加到历史记录

**历史记录页：**

- 左侧列表展示所有核验记录（带颜色标识：🟢绿色=正常 / 🔴红色=有破损 / 🟠橙色=有误检）
- 点击记录查看详情：现场截图 → 统计摘要（检测/破损/确认/误检/不确定）→ 目标详情表格 → 综合评估 → 维护建议 → 文件路径
- 支持横向滚动查看完整列表文字

### 2. Web 仪表盘

通过 `--web` 参数启动 Flask 服务，PC 浏览器远程查看：

| 页面 | 路径 | 功能 |
|------|------|------|
| 实时监控 | `/` | MJPEG 视频流（带检测框标注）+ 实时统计 + 核验状态 + 最新历史记录预览 |
| 核验结果 | `/results` | 分页展示全部核验记录，点击查看完整详情（截图、逐项核验、综合评估） |

API 接口：

| 接口 | 说明 |
|------|------|
| `/api/status` | 当前检测状态 (JSON) |
| `/api/history?limit=N` | 最近 N 条核验记录 |
| `/api/history/<id>` | 单条核验详情 |
| `/api/snapshot/<id>` | 核验截图文件 |
| `/video_feed` | MJPEG 实时视频流 |

### 3. LLM 智能核验

核验功能使用大模型对 NPU 检测结果进行二次确认。**所有检测到的绝缘子目标均会触发核验**，由 LLM 视觉判断是否真正存在破损。

| API | 模型 | 方式 | 环境变量 |
|-----|------|------|----------|
| 智谱 GLM-4V | `glm-4v` | 多模态视觉核验（直接分析图片） | `ZHIPU_API_KEY` |

API Key 获取：https://open.bigmodel.cn （免费注册）

核验返回结构化 JSON：

```json
{
  "verifications": [{
    "target_index": 1,
    "verification": "correct|incorrect|uncertain",
    "severity": "none|minor|moderate|severe|critical",
    "defect_description": "缺陷描述",
    "maintenance_suggestion": "维护建议"
  }],
  "overall_assessment": "整体评估"
}
```

核验有 90 秒超时保护，超时自动释放。

### 4. 大框分离逻辑

模型偶尔会检出一个包住整串绝缘子的大包围框。系统会自动识别这种大框并分离处理：

- **大框**：在画面中以黄色虚线绘制，标注"绝缘子串"，仅用于可视化
- **小框**（独立绝缘子）：正常参与核验、统计、记录

识别条件：框面积 > 4 倍中位面积，且包含至少 2 个其他框的中心点。

### 5. 数据记录

| 类型 | 保存路径 | 触发方式 |
|------|---------|----------|
| 截图 | `records/snapshots/snap_*.jpg` | 检测到绝缘子自动触发（1秒冷却）+ 手动触发 |
| 录像 | `records/video_YYYYmmdd_HHMMSS.mp4` | 手动开启 / 停止（H.264 avc1 编码） |
| 事件日志 | `records/events_YYYYmmdd_HHMMSS.json` | 录像期间检测到非 class-0 目标时自动记录 |
| 核验报告 | `records/reports/report_*.json` | LLM 核验完成后自动保存 |

---

## 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--camera` | `/dev/video11` | 摄像头设备路径 |
| `--width` | `1920` | 采集分辨率宽度 |
| `--height` | `1080` | 采集分辨率高度 |
| `--fps` | `15` | 采集帧率 |
| `--model` | `model.rknn` | RKNN 模型文件路径 |
| `--conf` | `0.25` | 通用置信度阈值（越低框越多，越高误检越少） |
| `--defect-conf` | `0.25` | 缺陷类 (class 2) 专用置信度阈值 |
| `--iou` | `0.45` | NMS IoU 阈值 |
| `--input-size` | `640` | 模型输入尺寸（像素） |
| `--min-area` | `8000` | 最小检测框面积（像素），过滤过小误检，0 表示关闭 |
| `--record` | `False` | 启动时自动开始录像 |
| `--output-dir` | `records` | 截图/录像/报告输出目录 |
| `--verify-api` | (空) | LLM 核验 API: `zhipu` |
| `--web` | `False` | 启动 Web 仪表盘 |
| `--web-port` | `5000` | Web 仪表盘端口号 |
| `--test-dir` | (空) | 测试模式：从此目录随机抽取 5 张图片检测核验，不走摄像头 |

---

## 项目结构

```
project/
├── main.py                    # 系统主入口 (Qt 界面 + NPU 推理 + LLM 核验 + 数据记录)
├── web_server.py              # Web 仪表盘 (Flask, MJPEG 推流 + 历史结果 API)
├── requirements.txt           # Python 依赖清单
├── README.md                  # 本文件
│
├── train/                     # 训练模块 (PC 端)
│   ├── prepare_dataset.py     #   数据集准备
│   ├── train_yolo.py          #   YOLO 模型训练
│   ├── extract_metrics.py     #   提取评估指标
│   ├── mine_hard_examples.py  #   难例挖掘
│   └── data_quality_check.py  #   数据质量检查
│
├── convert/                   # 模型转换模块 (PC 端)
│   ├── export_onnx.py         #   PyTorch → ONNX
│   ├── build_rknn.py          #   ONNX → RKNN
│   ├── compare_onnx_rknn.py   #   ONNX vs RKNN 精度对比
│   └── make_calib_list.py     #   生成量化校准列表
│
├── deploy/                    # 部署辅助脚本 (PC/板端)
│   ├── inference_display.py   #   OpenCV 桌面推理显示
│   ├── verify_board.py        #   板端环境验证
│   ├── tune_thresholds.py     #   阈值调优
│   ├── mvp_display.py         #   MVP 简易显示
│   └── realtime_simulator.py  #   实时模拟器
│
└── tools/                     # 工具脚本
    ├── pack_for_cloud.py      #   云端打包
    └── pack_elf2_for_cloud.py #   ELF2 云端打包
```

---

## 完整工作流

### 第一阶段：训练（PC）

```bash
cd project/train
python3 prepare_dataset.py --image-root /path/to/images --annotation-json /path/to/annotations.json
python3 train_yolo.py train --data ../data/yolo_dataset/dataset.yaml --model yolov8n.pt --epochs 100
```

### 第二阶段：模型转换（PC）

```bash
cd project/convert
python3 export_onnx.py --weights ../train/runs/detect/train/weights/best.pt --imgsz 640
python3 build_rknn.py --onnx best.onnx --output model.rknn
```

### 第三阶段：板端部署

```bash
# 拷贝 model.rknn 到板端
scp model.rknn elf@<板端IP>:/home/elf/RK3588/project/

# 板端运行
python3 main.py
```

---

## 常见问题

<details>
<summary><b>Q: 摄像头打不开？</b></summary>

```bash
# 查看可用摄像头
ls /dev/video*
v4l2-ctl --list-devices

# 通常 OV13855 在 /dev/video11 (rkisp_mainpath)
python3 main.py --camera /dev/video11
```
</details>

<details>
<summary><b>Q: Qt 界面无法启动？</b></summary>

```bash
# 确认 PyQt5 已安装
python3 -c "from PyQt5.QtWidgets import QApplication; print('OK')"

# 确认 Wayland 环境
ls /run/user/1000/wayland-0 && echo "Wayland OK"

# 如果报权限错误
sudo usermod -a -G render $USER
# 重新登录后生效
```
</details>

<details>
<summary><b>Q: 如何调整检测灵敏度？</b></summary>

```bash
# 降低 conf = 更多检测框 (可能误检增多)
python3 main.py --conf 0.15

# 提高 conf = 更少检测框 (可能漏检)
python3 main.py --conf 0.4

# 调低缺陷类专用阈值 (更多框标红)
python3 main.py --defect-conf 0.15

# 过滤过小误检框
python3 main.py --min-area 12000
```
</details>

<details>
<summary><b>Q: LLM 核验报错？</b></summary>

确认 API Key 已设置：
```bash
echo $ZHIPU_API_KEY
```

如果为空，先注册获取 Key，然后：
```bash
export ZHIPU_API_KEY="xxx.xxx"
python3 main.py --verify-api zhipu
```
</details>

<details>
<summary><b>Q: Web 仪表盘打不开？</b></summary>

```bash
# 确认 Flask 已安装
python3 -c "import flask; print(flask.__version__)"

# 确认板端 IP 地址
ip addr show wlan0 | grep inet

# 确认 PC 和板端在同一网络
ping <板端IP>

# 防火墙放行端口 (如有)
sudo ufw allow 5000
```
</details>

---

**硬件平台**：瑞芯微 RK3588 &nbsp;|&nbsp; **系统**：Linux aarch64 &nbsp;|&nbsp; **Python**：3.10 &nbsp;|&nbsp; **NPU**：6 TOPS

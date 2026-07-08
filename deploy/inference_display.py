import argparse
import json
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from rknnlite.api import RKNNLite


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float = 0.45):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
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
        inds = np.where(iou <= iou_thres)[0]
        order = order[inds + 1]
    return keep


def letterbox(img: np.ndarray, new_shape=(640, 640), color=(114, 114, 114)):
    shape = img.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)


def load_class_map(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def draw_status(frame, fps):
    cv2.putText(frame, f"FPS: {fps:.2f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    cv2.putText(frame, time.strftime("%Y-%m-%d %H:%M:%S"), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)


def draw_detections(frame, detections, class_map):
    for d in detections:
        x1, y1, x2, y2, score, cls = d
        cls = int(cls)
        color = (0, 255, 0) if cls == 0 else (0, 0, 255)
        label = f"{class_map.get(cls, str(cls))} {score:.2f}"
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(frame, label, (int(x1), max(0, int(y1) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def decode_yolov8_single_output(t: np.ndarray, conf_thres: float, iou_thres: float, orig_w: int, orig_h: int):
    # Supports output shaped like [1, 84, 8400] or [1, 8400, 84]
    if t.ndim == 3:
        t = t[0]
    if t.shape[0] < t.shape[1]:
        t = t.transpose(1, 0)
    if t.shape[1] < 6:
        return []

    boxes_xywh = t[:, :4]
    cls_scores = t[:, 4:]
    cls_id = np.argmax(cls_scores, axis=1)
    cls_conf = cls_scores[np.arange(len(cls_scores)), cls_id]

    mask = cls_conf >= conf_thres
    if not np.any(mask):
        return []

    boxes_xywh = boxes_xywh[mask]
    cls_id = cls_id[mask]
    cls_conf = cls_conf[mask]

    x, y, w, h = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
    x1 = np.clip(x - w / 2, 0, orig_w - 1)
    y1 = np.clip(y - h / 2, 0, orig_h - 1)
    x2 = np.clip(x + w / 2, 0, orig_w - 1)
    y2 = np.clip(y + h / 2, 0, orig_h - 1)
    boxes = np.stack([x1, y1, x2, y2], axis=1)

    keep = nms(boxes, cls_conf, iou_thres=iou_thres)
    out = []
    for i in keep:
        out.append([boxes[i, 0], boxes[i, 1], boxes[i, 2], boxes[i, 3], float(cls_conf[i]), int(cls_id[i])])
    return out


def postprocess(outputs, conf_thres=0.25, iou_thres=0.45, orig_w=1920, orig_h=1080):
    if not outputs:
        return []
    # Try to decode the largest tensor as YOLO head.
    tensors = [np.asarray(o) for o in outputs]
    tensors.sort(key=lambda x: x.size, reverse=True)
    for t in tensors:
        dets = decode_yolov8_single_output(t, conf_thres, iou_thres, orig_w, orig_h)
        if dets:
            return dets
    return []


def run(args):
    class_map = load_class_map(Path(args.class_map))
    image_dir = Path(args.image_dir)
    image_paths = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])
    if not image_paths:
        raise RuntimeError(f"No images found in {image_dir}")

    rknn = RKNNLite()
    if rknn.load_rknn(args.model) != 0:
        raise RuntimeError("Failed to load RKNN model.")
    if rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2) != 0:
        raise RuntimeError("Failed to init RKNN runtime.")

    fps_window = deque(maxlen=20)
    idx = 0
    delay_ms = int(1000 / max(1, args.fps))

    while True:
        t0 = time.time()
        frame = cv2.imread(str(image_paths[idx]))
        idx = (idx + 1) % len(image_paths)
        if frame is None:
            continue

        inp, _, _ = letterbox(frame, (args.input_size, args.input_size))
        inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
        inp = np.expand_dims(inp, axis=0)

        outputs = rknn.inference(inputs=[inp])
        detections = postprocess(
            outputs,
            conf_thres=args.conf,
            iou_thres=args.iou,
            orig_w=frame.shape[1],
            orig_h=frame.shape[0],
        )
        draw_detections(frame, detections, class_map)

        fps_window.append(1.0 / max(1e-6, time.time() - t0))
        fps = sum(fps_window) / len(fps_window)
        draw_status(frame, fps)

        cv2.imshow(args.window_name, frame)
        key = cv2.waitKey(delay_ms) & 0xFF
        if key == ord("q") or key == 27:
            break

    rknn.release()
    cv2.destroyAllWindows()


def parse_args():
    p = argparse.ArgumentParser(description="Pseudo real-time display on ELF2 using RKNN.")
    p.add_argument("--model", default="project/convert/model.rknn")
    p.add_argument("--image-dir", default="project/data/stream_images")
    p.add_argument("--class-map", default="project/assets/class_map.json")
    p.add_argument("--input-size", type=int, default=640)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--window-name", default="ELF2 Insulator Defect Detection")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())

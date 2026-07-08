import argparse
import json
import time
from collections import Counter, deque
from pathlib import Path

import cv2
from ultralytics import YOLO


def load_class_map(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def list_images(image_dir: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted([p for p in image_dir.iterdir() if p.suffix.lower() in exts])


def draw_summary(frame, counts: Counter, class_map: dict):
    y = frame.shape[0] - 20
    parts = [f"{class_map.get(k, str(k))}:{v}" for k, v in sorted(counts.items()) if v > 0]
    text = " | ".join(parts) if parts else "未检测到目标"
    cv2.rectangle(frame, (0, y - 28), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
    cv2.putText(frame, text, (12, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)


def run(args):
    model = YOLO(args.weights)
    class_map = load_class_map(Path(args.class_map))
    image_dir = Path(args.image_dir)
    paths = list_images(image_dir)
    if not paths:
        raise RuntimeError(f"No images found in {image_dir}")

    fps_hist = deque(maxlen=30)
    delay = int(1000 / max(1, args.fps))
    idx = 0

    window = args.window_name
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    if args.fullscreen:
        cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print(f"Loaded model: {args.weights}")
    print(f"Images: {len(paths)} from {image_dir}")
    print("Press q or ESC to exit.")

    while True:
        t0 = time.time()
        frame = cv2.imread(str(paths[idx]))
        idx = (idx + 1) % len(paths)
        if frame is None:
            continue

        result = model.predict(
            frame,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            verbose=False,
        )[0]

        counts = Counter()
        boxes = result.boxes
        if boxes is not None:
            for b in boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                conf = float(b.conf[0])
                cls_id = int(b.cls[0])
                counts[cls_id] += 1
                label = f"{class_map.get(cls_id, str(cls_id))} {conf:.2f}"
                color = (0, 220, 0) if cls_id == 0 else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    label,
                    (x1, max(24, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                )

        fps_hist.append(1.0 / max(1e-6, time.time() - t0))
        fps = sum(fps_hist) / len(fps_hist)
        cv2.putText(frame, f"FPS {fps:.1f}", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(
            frame,
            time.strftime("%Y-%m-%d %H:%M:%S"),
            (16, 64),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )
        draw_summary(frame, counts, class_map)

        cv2.imshow(window, frame)
        key = cv2.waitKey(delay) & 0xFF
        if key in (27, ord("q")):
            break

    cv2.destroyAllWindows()


def parse_args():
    p = argparse.ArgumentParser(description="Fast MVP: insulator defect detection on external display.")
    p.add_argument("--weights", required=True, help="Path to best.pt")
    p.add_argument("--image-dir", default="data/stream_images")
    p.add_argument("--class-map", default="assets/class_map_zh.json")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--fullscreen", action="store_true")
    p.add_argument("--window-name", default="绝缘子缺陷检测")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())

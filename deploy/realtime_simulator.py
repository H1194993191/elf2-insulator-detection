import argparse
import json
import time
from collections import deque
from pathlib import Path

import cv2
from ultralytics import YOLO


def load_class_map(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def run(args):
    model = YOLO(args.weights)
    class_map = load_class_map(Path(args.class_map))
    image_dir = Path(args.image_dir)
    paths = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])
    if not paths:
        raise RuntimeError(f"No images under {image_dir}")

    idx = 0
    fps_hist = deque(maxlen=30)
    delay = int(1000 / max(1, args.fps))

    while True:
        t0 = time.time()
        frame = cv2.imread(str(paths[idx]))
        idx = (idx + 1) % len(paths)
        if frame is None:
            continue

        result = model.predict(frame, imgsz=args.imgsz, conf=args.conf, iou=args.iou, verbose=False)[0]
        boxes = result.boxes
        if boxes is not None:
            for b in boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                conf = float(b.conf[0])
                cls_id = int(b.cls[0])
                label = f"{class_map.get(cls_id, str(cls_id))} {conf:.2f}"
                color = (0, 255, 0) if cls_id == 0 else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        fps_hist.append(1.0 / max(1e-6, time.time() - t0))
        fps = sum(fps_hist) / len(fps_hist)
        cv2.putText(frame, f"FPS {fps:.2f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(frame, time.strftime("%Y-%m-%d %H:%M:%S"), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow(args.window_name, frame)
        key = cv2.waitKey(delay) & 0xFF
        if key in (27, ord("q")):
            break

    cv2.destroyAllWindows()


def parse_args():
    p = argparse.ArgumentParser(description="Pseudo real-time inference simulator with image sequence.")
    p.add_argument("--weights", required=True, help="Path to best.pt")
    p.add_argument("--image-dir", default="project/data/stream_images")
    p.add_argument("--class-map", default="project/assets/class_map.json")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--window-name", default="ELF2 Defect Monitor (Simulator)")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())

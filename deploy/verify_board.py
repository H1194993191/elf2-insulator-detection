"""Quick check: can best.pt run on ELF2? Single-image inference, no display needed."""
import argparse
import json
import sys
from pathlib import Path

import cv2


def load_class_map(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def main():
    parser = argparse.ArgumentParser(description="Verify YOLO weights on board (one image).")
    parser.add_argument("--weights", default="models/best.pt")
    parser.add_argument("--image", required=True, help="One test image path")
    parser.add_argument("--class-map", default="assets/class_map_elf2_zh.json")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    weights = Path(args.weights)
    image = Path(args.image)
    if not weights.exists():
        print(f"FAIL: weights not found: {weights}")
        sys.exit(1)
    if not image.exists():
        print(f"FAIL: image not found: {image}")
        sys.exit(1)

    print(f"Loading model: {weights}")
    from ultralytics import YOLO

    model = YOLO(str(weights))
    class_map = load_class_map(Path(args.class_map))

    print(f"Inference: {image}")
    result = model.predict(str(image), imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        print("OK: model runs, but no boxes detected (try lower --conf 0.15)")
        sys.exit(0)

    print(f"OK: detected {len(boxes)} object(s)")
    for i, b in enumerate(boxes):
        cls_id = int(b.cls[0])
        conf = float(b.conf[0])
        name = class_map.get(cls_id, str(cls_id))
        xyxy = [round(v, 1) for v in b.xyxy[0].tolist()]
        print(f"  [{i}] {name} {conf:.2f}  box={xyxy}")

    out = Path("runs/verify_out.jpg")
    out.parent.mkdir(parents=True, exist_ok=True)
    annotated = result.plot()
    cv2.imwrite(str(out), annotated)
    print(f"Saved: {out.resolve()}")


if __name__ == "__main__":
    main()

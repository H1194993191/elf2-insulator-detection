import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Mine hard examples based on low-confidence detections.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output-json", default="project/data/hard_examples.json")
    parser.add_argument("--conf-low", type=float, default=0.15)
    parser.add_argument("--conf-high", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    model = YOLO(args.weights)
    image_paths = sorted([p for p in Path(args.image_dir).rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])
    hard = []

    for p in image_paths:
        r = model.predict(str(p), imgsz=args.imgsz, conf=args.conf_low, verbose=False)[0]
        uncertain = False
        if r.boxes is not None:
            for c in r.boxes.conf.tolist():
                if args.conf_low <= float(c) <= args.conf_high:
                    uncertain = True
                    break
        if uncertain:
            hard.append(str(p.as_posix()))

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"hard_examples": hard}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Hard examples: {len(hard)} written to {out}")


if __name__ == "__main__":
    main()

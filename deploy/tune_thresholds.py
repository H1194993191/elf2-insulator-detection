import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def evaluate_at_threshold(model, image_paths, conf, iou, imgsz):
    # Lightweight proxy score: average detections per frame and confidence.
    # For production, replace with proper label-aware evaluation.
    det_count = 0
    conf_sum = 0.0
    for p in image_paths:
        r = model.predict(str(p), conf=conf, iou=iou, imgsz=imgsz, verbose=False)[0]
        boxes = r.boxes
        if boxes is None:
            continue
        c = boxes.conf.tolist()
        det_count += len(c)
        conf_sum += sum(float(v) for v in c)
    n = max(1, len(image_paths))
    return {
        "avg_det_per_image": det_count / n,
        "avg_conf": conf_sum / max(1, det_count),
    }


def main():
    parser = argparse.ArgumentParser(description="Grid search for inference thresholds.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output-json", default="project/deploy/threshold_sweep.json")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    model = YOLO(args.weights)
    image_paths = sorted([p for p in Path(args.image_dir).rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])
    if not image_paths:
        raise RuntimeError("No images for threshold tuning")

    conf_list = [0.15, 0.2, 0.25, 0.3, 0.35]
    iou_list = [0.4, 0.45, 0.5]

    rows = []
    for conf in conf_list:
        for iou in iou_list:
            stat = evaluate_at_threshold(model, image_paths, conf, iou, args.imgsz)
            rows.append({"conf": conf, "iou": iou, **stat})
            print(rows[-1])

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Threshold sweep saved to {out}")


if __name__ == "__main__":
    main()

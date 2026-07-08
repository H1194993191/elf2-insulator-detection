import argparse
from pathlib import Path

from ultralytics import YOLO


def train(args: argparse.Namespace) -> None:
    model = YOLO(args.model)
    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        cache=args.cache,
        pretrained=True,
    )

    best = Path(args.project) / args.name / "weights" / "best.pt"
    print(f"Training complete. Best checkpoint: {best}")


def validate(args: argparse.Namespace) -> None:
    model = YOLO(args.weights)
    metrics = model.val(data=args.data, imgsz=args.imgsz, conf=args.conf, iou=args.iou, device=args.device)
    print("Validation done.")
    print(metrics)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train and validate YOLO model for insulator defects.")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--model", default="yolov8n.pt")
    t.add_argument("--data", default="project/data/yolo_dataset/dataset.yaml")
    t.add_argument("--imgsz", type=int, default=640)
    t.add_argument("--epochs", type=int, default=100)
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--device", default="0")
    t.add_argument("--workers", type=int, default=8)
    t.add_argument("--project", default="project/runs/train")
    t.add_argument("--name", default="insulator_v1")
    t.add_argument("--cache", action="store_true")

    v = sub.add_parser("val")
    v.add_argument("--weights", required=True)
    v.add_argument("--data", default="project/data/yolo_dataset/dataset.yaml")
    v.add_argument("--imgsz", type=int, default=640)
    v.add_argument("--conf", type=float, default=0.25)
    v.add_argument("--iou", type=float, default=0.45)
    v.add_argument("--device", default="0")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.cmd == "train":
        train(args)
    elif args.cmd == "val":
        validate(args)

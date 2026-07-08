import argparse
from pathlib import Path

from ultralytics import YOLO


def export_onnx(weights: str, imgsz: int, opset: int, simplify: bool) -> Path:
    model = YOLO(weights)
    out = model.export(format="onnx", imgsz=imgsz, opset=opset, simplify=simplify, dynamic=False)
    return Path(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export YOLO checkpoint to ONNX.")
    parser.add_argument("--weights", required=True, help="Path to best.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--simplify", action="store_true")
    args = parser.parse_args()

    onnx_path = export_onnx(args.weights, args.imgsz, args.opset, args.simplify)
    print(f"ONNX exported: {onnx_path}")


if __name__ == "__main__":
    main()

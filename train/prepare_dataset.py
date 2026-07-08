import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple


CLASS_TO_ID = {
    "normal": 0,
    "broken": 1,
    "pollution": 2,
    "flashover_trace": 3,
    "foreign_object": 4,
}


def load_annotations(annotation_path: Path) -> Dict[str, List[dict]]:
    with annotation_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return raw


def to_yolo_line(box: List[float], class_id: int, width: int, height: int) -> str:
    x1, y1, x2, y2 = box
    x_center = ((x1 + x2) / 2.0) / width
    y_center = ((y1 + y2) / 2.0) / height
    w = (x2 - x1) / width
    h = (y2 - y1) / height
    return f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"


def split_items(items: List[str], seed: int, ratios: Tuple[float, float, float]) -> Dict[str, List[str]]:
    random.Random(seed).shuffle(items)
    n = len(items)
    train_end = int(n * ratios[0])
    val_end = train_end + int(n * ratios[1])
    return {
        "train": items[:train_end],
        "val": items[train_end:val_end],
        "test": items[val_end:],
    }


def ensure_dirs(output_root: Path) -> None:
    for split in ("train", "val", "test"):
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)


def write_dataset_yaml(output_root: Path) -> None:
    yaml_path = output_root / "dataset.yaml"
    yaml_text = """path: .
train: images/train
val: images/val
test: images/test

names:
  0: normal
  1: broken
  2: pollution
  3: flashover_trace
  4: foreign_object
"""
    yaml_path.write_text(yaml_text, encoding="utf-8")


def convert(args: argparse.Namespace) -> None:
    image_root = Path(args.image_root)
    output_root = Path(args.output_root)
    annotation_map = load_annotations(Path(args.annotation_json))

    ensure_dirs(output_root)
    image_names = sorted(annotation_map.keys())
    split_map = split_items(image_names, args.seed, (0.7, 0.2, 0.1))

    for split, names in split_map.items():
        for image_name in names:
            src_img = image_root / image_name
            dst_img = output_root / "images" / split / image_name
            shutil.copy2(src_img, dst_img)

            anns = annotation_map.get(image_name, [])
            width = anns[0]["image_width"] if anns else args.default_width
            height = anns[0]["image_height"] if anns else args.default_height
            label_lines = []
            for ann in anns:
                class_name = ann["class_name"]
                if class_name not in CLASS_TO_ID:
                    continue
                label_lines.append(
                    to_yolo_line(
                        ann["bbox_xyxy"],
                        CLASS_TO_ID[class_name],
                        ann.get("image_width", width),
                        ann.get("image_height", height),
                    )
                )
            label_path = output_root / "labels" / split / f"{Path(image_name).stem}.txt"
            label_path.write_text("\n".join(label_lines), encoding="utf-8")

    write_dataset_yaml(output_root)
    print(f"Done. YOLO dataset prepared at: {output_root}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prepare YOLO dataset from JSON annotations.")
    p.add_argument("--image-root", required=True, help="Folder containing source images.")
    p.add_argument("--annotation-json", required=True, help="JSON file with annotation map.")
    p.add_argument("--output-root", default="project/data/yolo_dataset", help="Output dataset root.")
    p.add_argument("--seed", type=int, default=42, help="Random seed for split.")
    p.add_argument("--default-width", type=int, default=1920, help="Fallback image width.")
    p.add_argument("--default-height", type=int, default=1080, help="Fallback image height.")
    return p


if __name__ == "__main__":
    parser = build_arg_parser()
    convert(parser.parse_args())

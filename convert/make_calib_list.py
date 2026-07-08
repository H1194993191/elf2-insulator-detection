import argparse
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Create RKNN calibration list text file.")
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output", default="project/convert/calib_list.txt")
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    files = [p for p in image_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    if not files:
        raise RuntimeError("No images found for calibration.")
    random.Random(args.seed).shuffle(files)
    chosen = files[: min(args.count, len(files))]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(str(p.as_posix()) for p in chosen), encoding="utf-8")
    print(f"Calibration file generated: {out} ({len(chosen)} images)")


if __name__ == "__main__":
    main()

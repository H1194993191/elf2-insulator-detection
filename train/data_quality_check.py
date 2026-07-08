import argparse
from collections import Counter
from pathlib import Path


def read_label_file(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        cols = line.split()
        if len(cols) != 5:
            continue
        rows.append(cols)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Basic YOLO label quality checks.")
    parser.add_argument("--dataset-root", default="project/data/yolo_dataset")
    args = parser.parse_args()

    root = Path(args.dataset_root)
    counter = Counter()
    empty_labels = 0
    broken_rows = 0

    for split in ("train", "val", "test"):
        label_dir = root / "labels" / split
        for label in label_dir.glob("*.txt"):
            rows = read_label_file(label)
            if not rows:
                empty_labels += 1
                continue
            for row in rows:
                try:
                    cls = int(float(row[0]))
                    vals = list(map(float, row[1:]))
                    if any(v < 0 or v > 1 for v in vals):
                        broken_rows += 1
                    counter[cls] += 1
                except ValueError:
                    broken_rows += 1

    print("Class count:", dict(counter))
    print("Empty labels:", empty_labels)
    print("Broken rows:", broken_rows)


if __name__ == "__main__":
    main()

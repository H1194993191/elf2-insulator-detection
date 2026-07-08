import argparse
import zipfile
from pathlib import Path


def add_tree(zf: zipfile.ZipFile, folder: Path, arc_root: str):
    for p in folder.rglob("*"):
        if p.is_file():
            arcname = Path(arc_root) / p.relative_to(folder)
            zf.write(p, arcname.as_posix())


def main():
    parser = argparse.ArgumentParser(description="Pack ELF2 dataset + project for cloud training.")
    parser.add_argument("--dataset", default="../elf2/低光照和无光照", help="YOLO dataset folder")
    parser.add_argument("--project", default=".", help="project folder")
    parser.add_argument("--output", default="elf2_cloud_train.zip")
    args = parser.parse_args()

    dataset = Path(args.dataset).resolve()
    project = Path(args.project).resolve()
    output = Path(args.output).resolve()

    if not (dataset / "dataset.yaml").exists():
        raise RuntimeError(f"Missing dataset.yaml in {dataset}")

    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zf:
        add_tree(zf, dataset, "dataset")
        for sub in ("train", "deploy", "assets", "tools"):
            src = project / sub
            if src.exists():
                add_tree(zf, src, f"project/{sub}")
        req = project / "requirements-train.txt"
        if req.exists():
            zf.write(req, "project/requirements-train.txt")

    print(f"Created: {output}")
    print("Upload to cloud, then:")
    print("  unzip elf2_cloud_train.zip")
    print("  pip install -r project/requirements-train.txt")
    print("  python project/train/train_yolo.py train --data dataset/dataset.yaml ...")


if __name__ == "__main__":
    main()

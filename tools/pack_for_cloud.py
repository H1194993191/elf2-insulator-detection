import argparse
import zipfile
from pathlib import Path


def add_dir(zf: zipfile.ZipFile, folder: Path, arc_prefix: str):
    if not folder.exists():
        print(f"Skip missing folder: {folder}")
        return
    for p in folder.rglob("*"):
        if p.is_file():
            rel = p.relative_to(folder.parent.parent if arc_prefix == "project" else folder.parent)
            if arc_prefix == "project":
                arcname = Path("project") / p.relative_to(folder.parent)
            else:
                arcname = Path("project/data") / p.relative_to(folder)
            zf.write(p, arcname.as_posix())


def main():
    parser = argparse.ArgumentParser(description="Pack project files for cloud GPU training.")
    parser.add_argument("--project-root", default=".", help="Path to project folder")
    parser.add_argument("--data-root", default="data", help="Path to data folder under project")
    parser.add_argument("--output", default="insulator_cloud.zip")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    data_root = project_root / args.data_root
    output = Path(args.output).resolve()

    include_dirs = ["train", "convert", "assets", "tools"]
    include_files = ["requirements-train.txt"]

    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zf:
        for d in include_dirs:
            src = project_root / d
            if not src.exists():
                continue
            for p in src.rglob("*"):
                if p.is_file():
                    zf.write(p, Path("project") / p.relative_to(project_root).as_posix())

        for f in include_files:
            src = project_root / f
            if src.exists():
                zf.write(src, Path("project") / f)

        if data_root.exists():
            for p in data_root.rglob("*"):
                if p.is_file():
                    zf.write(p, Path("project/data") / p.relative_to(data_root).as_posix())

    print(f"Packed: {output}")
    print("Upload this zip to cloud, unzip, then follow project/docs/CLOUD_TRAIN.md")


if __name__ == "__main__":
    main()

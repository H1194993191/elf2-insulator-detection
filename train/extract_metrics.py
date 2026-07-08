import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Extract key metrics from Ultralytics results.csv")
    parser.add_argument("--results-csv", required=True, help="Path to results.csv")
    parser.add_argument("--output", default="project/docs/train_report.md")
    args = parser.parse_args()

    csv_path = Path(args.results_csv)
    rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        raise RuntimeError("No rows in results.csv")

    last = rows[-1]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    report = f"""# Training Report

- Epoch: {last.get('epoch', 'N/A')}
- mAP50: {last.get('metrics/mAP50(B)', 'N/A')}
- mAP50-95: {last.get('metrics/mAP50-95(B)', 'N/A')}
- Precision: {last.get('metrics/precision(B)', 'N/A')}
- Recall: {last.get('metrics/recall(B)', 'N/A')}

## Notes
- Source CSV: `{csv_path.as_posix()}`
- Update this report after each retraining cycle.
"""
    out.write_text(report, encoding="utf-8")
    print(f"Report written: {out}")


if __name__ == "__main__":
    main()

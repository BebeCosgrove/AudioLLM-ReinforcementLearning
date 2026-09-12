"""
summarize_results.py
--------------------
Collate every result file under results/ into one table.

Prints overall accuracy and domain-average accuracy per (model, split,
perturbation, alpha), with the delta against that model/split's own no-CD
baseline when one has been run. Optionally writes the same table as CSV.

Usage
-----
    python contrastive_decoding/summarize_results.py
    python contrastive_decoding/summarize_results.py --csv summary.csv
    python contrastive_decoding/summarize_results.py --dedup
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

RESULTS_ROOT = Path(__file__).resolve().parent / "results"


def load_runs(dedup: bool):
    """Yield one row per result file found under results/<model>/<split>/."""
    rows = []
    for path in sorted(RESULTS_ROOT.glob("*/*/*.json")):
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)

        metadata = payload.get("metadata", {})
        block = payload.get("deduplicated", payload) if dedup else payload

        rows.append(
            {
                "model": metadata.get("model", path.parent.parent.name),
                "split": metadata.get("split", path.parent.name),
                "perturbation": metadata.get("perturbation") or "-",
                "alpha": metadata.get("alpha"),
                "accuracy": block["overall"]["accuracy"],
                "domain_average": block["domain_average_accuracy"],
                "correct": block["overall"]["correct"],
                "total": block["overall"]["total"],
                "file": path.name,
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dedup",
        action="store_true",
        help="Report the padding-free numbers instead of the accelerate-padded ones.",
    )
    parser.add_argument("--csv", default=None, help="Also write the table to this path.")
    args = parser.parse_args()

    rows = load_runs(args.dedup)
    if not rows:
        print(f"No result files found under {RESULTS_ROOT}")
        return

    baselines = {
        (row["model"], row["split"]): row
        for row in rows
        if row["perturbation"] == "-"
    }

    label = "deduplicated" if args.dedup else "as-run (includes accelerate padding)"
    print(f"Reporting: {label}\n")

    header = (
        f"{'model':6s} {'split':11s} {'perturbation':13s} {'alpha':>5s} "
        f"{'acc':>7s} {'d-acc':>7s} {'vs base':>9s} {'n':>6s}"
    )
    print(header)
    print("-" * len(header))

    current = None
    for row in sorted(
        rows,
        key=lambda r: (r["model"], r["split"], r["perturbation"], r["alpha"] or 0),
    ):
        group = (row["model"], row["split"])
        if group != current:
            if current is not None:
                print()
            current = group

        baseline = baselines.get(group)
        if baseline is not None and row["perturbation"] != "-":
            delta = f"{row['accuracy'] - baseline['accuracy']:+.4f}"
        else:
            delta = "-"

        alpha = "-" if row["alpha"] is None else f"{row['alpha']:.1f}"
        print(
            f"{row['model']:6s} {row['split']:11s} {row['perturbation']:13s} {alpha:>5s} "
            f"{row['accuracy']:7.4f} {row['domain_average']:7.4f} {delta:>9s} "
            f"{row['total']:6d}"
        )

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()

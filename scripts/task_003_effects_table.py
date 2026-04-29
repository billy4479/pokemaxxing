"""Task 003: build a unified effects table from CSVs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.dataset.queries import build_effects_table, build_removed_effects_table


def run(data_dir: str = "csv", out: str = ""):
    """Return the filtered effects table and optionally persist to CSV."""
    effects_df = build_effects_table(data_dir=data_dir)

    if out:
        output_path = Path(out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        effects_df.to_csv(output_path, index=False)

    return effects_df


def run_removed(data_dir: str = "csv", out: str = ""):
    """Return removed effects table and optionally persist to CSV."""
    removed_df = build_removed_effects_table(data_dir=data_dir)

    if out:
        output_path = Path(out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        removed_df.to_csv(output_path, index=False)

    return removed_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Task 003 - filtered + removed effects tables"
    )
    parser.add_argument(
        "--data-dir", default="csv", help="Directory containing CSV files"
    )
    parser.add_argument("--out", default="", help="Optional filtered output CSV path")
    parser.add_argument(
        "--removed-out",
        default="",
        help="Optional removed effects output CSV path",
    )
    parser.add_argument("--rows", type=int, default=20, help="Rows to print")
    args = parser.parse_args()

    effects_df = run(data_dir=args.data_dir, out=args.out)
    removed_df = run_removed(data_dir=args.data_dir, out=args.removed_out)

    print(f"Filtered effects rows: {len(effects_df)}")
    print(f"Removed effects rows: {len(removed_df)}")
    print(effects_df.head(max(args.rows, 0)).to_string(index=False))


if __name__ == "__main__":
    main()

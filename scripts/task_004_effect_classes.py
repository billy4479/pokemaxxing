"""Task 004: parse filtered effects into typed effect classes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.dataset.queries import build_effect_classes_df


def run(data_dir: str = "csv", out: str = "", filtered: bool = True):
    """Return the parsed effect classes DataFrame and optionally persist it."""
    classes_df = build_effect_classes_df(data_dir=data_dir, filtered=filtered)

    if out:
        output_path = Path(out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        classes_df.to_csv(output_path, index=False)

    return classes_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 004 - parse effect classes")
    parser.add_argument(
        "--data-dir", default="csv", help="Directory containing CSV files"
    )
    parser.add_argument("--out", default="", help="Optional output CSV path")
    parser.add_argument(
        "--unfiltered",
        action="store_true",
        help="Parse unfiltered full effects table instead of filtered effects",
    )
    parser.add_argument("--rows", type=int, default=20, help="Rows to print")
    args = parser.parse_args()

    classes_df = run(data_dir=args.data_dir, out=args.out, filtered=not args.unfiltered)
    print(classes_df.head(max(args.rows, 0)).to_string(index=False))


if __name__ == "__main__":
    main()

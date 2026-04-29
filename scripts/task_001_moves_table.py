"""Task 001: build a unified moves table from CSVs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.dataset.queries import build_moves_table


def run(data_dir: str = "csv", out: str = ""):
    """Return the unified moves table and optionally persist to CSV."""
    moves_df = build_moves_table(data_dir=data_dir)

    if out:
        output_path = Path(out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        moves_df.to_csv(output_path, index=False)

    return moves_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 001 - unified moves table")
    parser.add_argument(
        "--data-dir", default="csv", help="Directory containing CSV files"
    )
    parser.add_argument("--out", default="", help="Optional output CSV path")
    parser.add_argument("--rows", type=int, default=10, help="Rows to print")
    args = parser.parse_args()

    moves_df = run(data_dir=args.data_dir, out=args.out)
    print(moves_df.head(max(args.rows, 0)).to_string(index=False))


if __name__ == "__main__":
    main()

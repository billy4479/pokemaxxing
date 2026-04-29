"""Task 005: rank effects by move coverage and learner coverage."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.dataset.queries import build_effect_ranking_table


def run(data_dir: str = "csv", out: str = "", filtered: bool = False):
    """Return ranked effects and optionally persist to CSV."""
    ranking_df = build_effect_ranking_table(data_dir=data_dir, filtered=filtered)

    if out:
        output_path = Path(out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ranking_df.to_csv(output_path, index=False)

    return ranking_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 005 - effect ranking")
    parser.add_argument(
        "--data-dir", default="csv", help="Directory containing CSV files"
    )
    parser.add_argument("--out", default="", help="Optional output CSV path")
    parser.add_argument("--rows", type=int, default=20, help="Rows to print")
    parser.add_argument(
        "--filtered",
        action="store_true",
        help="Rank filtered effects only (default: unfiltered)",
    )
    args = parser.parse_args()

    ranking_df = run(data_dir=args.data_dir, out=args.out, filtered=args.filtered)
    print(ranking_df.head(max(args.rows, 0)).to_string(index=False))


if __name__ == "__main__":
    main()

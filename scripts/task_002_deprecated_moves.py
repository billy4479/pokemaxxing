"""Task 002: infer moves deprecated in newer version groups."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.dataset.queries import (
    has_explicit_move_deprecation_flag,
    infer_deprecated_moves,
)


def run(data_dir: str = "csv", out: str = "", reference_version_group: str = ""):
    """Return inferred deprecated moves and optionally persist to CSV."""
    deprecated_df = infer_deprecated_moves(
        data_dir=data_dir,
        reference_version_group=(reference_version_group or None),
    )

    if out:
        output_path = Path(out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        deprecated_df.to_csv(output_path, index=False)

    return deprecated_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 002 - infer deprecated moves")
    parser.add_argument(
        "--data-dir", default="csv", help="Directory containing CSV files"
    )
    parser.add_argument(
        "--reference-version-group", default="", help="Version group identifier"
    )
    parser.add_argument("--out", default="", help="Optional output CSV path")
    parser.add_argument("--rows", type=int, default=20, help="Rows to print")
    args = parser.parse_args()

    explicit_flag = has_explicit_move_deprecation_flag(data_dir=args.data_dir)
    deprecated_df = run(
        data_dir=args.data_dir,
        out=args.out,
        reference_version_group=args.reference_version_group,
    )

    print(f"Explicit deprecation flag present: {explicit_flag}")
    print(deprecated_df.head(max(args.rows, 0)).to_string(index=False))


if __name__ == "__main__":
    main()

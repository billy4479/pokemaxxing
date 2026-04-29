"""CLI entrypoint for dataset queries."""

from __future__ import annotations

import argparse
from pathlib import Path

from .queries import (
    build_effect_classes_df,
    build_effect_ranking_table,
    build_effects_table,
    build_removed_effects_table,
    build_moves_table,
    has_explicit_move_deprecation_flag,
    infer_deprecated_moves,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pokemon dataset query CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    moves_parser = subparsers.add_parser(
        "moves-table", help="Build the unified moves table"
    )
    moves_parser.add_argument(
        "--data-dir", default="csv", help="Directory containing CSV files"
    )
    moves_parser.add_argument(
        "--out",
        default="",
        help="Optional output CSV path. If omitted, prints preview only.",
    )
    moves_parser.add_argument(
        "--rows",
        type=int,
        default=10,
        help="Number of rows to print as preview",
    )

    deprecated_parser = subparsers.add_parser(
        "deprecated-moves",
        help="Infer moves absent from a newer reference version group",
    )
    deprecated_parser.add_argument(
        "--data-dir", default="csv", help="Directory containing CSV files"
    )
    deprecated_parser.add_argument(
        "--reference-version-group",
        default="",
        help="Reference version_group identifier (default: latest represented in pokemon_moves)",
    )
    deprecated_parser.add_argument(
        "--out",
        default="",
        help="Optional output CSV path. If omitted, prints preview only.",
    )
    deprecated_parser.add_argument(
        "--rows",
        type=int,
        default=20,
        help="Number of rows to print as preview",
    )

    effects_parser = subparsers.add_parser(
        "effects-table", help="Build the unified effects table"
    )
    effects_parser.add_argument(
        "--data-dir", default="csv", help="Directory containing CSV files"
    )
    effects_parser.add_argument(
        "--out",
        default="",
        help="Optional output CSV path. If omitted, prints preview only.",
    )
    effects_parser.add_argument(
        "--rows",
        type=int,
        default=20,
        help="Number of rows to print as preview",
    )

    removed_effects_parser = subparsers.add_parser(
        "removed-effects-table",
        help="Build the complementary table of removed effects",
    )
    removed_effects_parser.add_argument(
        "--data-dir", default="csv", help="Directory containing CSV files"
    )
    removed_effects_parser.add_argument(
        "--out",
        default="",
        help="Optional output CSV path. If omitted, prints preview only.",
    )
    removed_effects_parser.add_argument(
        "--rows",
        type=int,
        default=20,
        help="Number of rows to print as preview",
    )

    effect_classes_parser = subparsers.add_parser(
        "effect-classes",
        help="Parse filtered effects into typed classes",
    )
    effect_classes_parser.add_argument(
        "--data-dir", default="csv", help="Directory containing CSV files"
    )
    effect_classes_parser.add_argument(
        "--out",
        default="",
        help="Optional output CSV path. If omitted, prints preview only.",
    )
    effect_classes_parser.add_argument(
        "--rows",
        type=int,
        default=20,
        help="Number of rows to print as preview",
    )
    effect_classes_parser.add_argument(
        "--unfiltered",
        action="store_true",
        help="Parse unfiltered full effects table instead of filtered effects",
    )

    ranking_parser = subparsers.add_parser(
        "effect-ranking",
        help="Rank effects by move_count * learner_count",
    )
    ranking_parser.add_argument(
        "--data-dir", default="csv", help="Directory containing CSV files"
    )
    ranking_parser.add_argument(
        "--out",
        default="",
        help="Optional output CSV path. If omitted, prints preview only.",
    )
    ranking_parser.add_argument(
        "--rows",
        type=int,
        default=20,
        help="Number of rows to print as preview",
    )
    ranking_parser.add_argument(
        "--filtered",
        action="store_true",
        help="Rank filtered effects only (default: unfiltered)",
    )

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "moves-table":
        moves_df = build_moves_table(data_dir=args.data_dir)
        if args.out:
            output_path = Path(args.out)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            moves_df.to_csv(output_path, index=False)
            print(f"Wrote {len(moves_df)} rows to {output_path}")

        preview_rows = max(args.rows, 0)
        if preview_rows:
            print(moves_df.head(preview_rows).to_string(index=False))

    if args.command == "deprecated-moves":
        explicit_flag = has_explicit_move_deprecation_flag(data_dir=args.data_dir)
        deprecated_df = infer_deprecated_moves(
            data_dir=args.data_dir,
            reference_version_group=(args.reference_version_group or None),
        )

        print(f"Explicit deprecation flag present: {explicit_flag}")
        if args.out:
            output_path = Path(args.out)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            deprecated_df.to_csv(output_path, index=False)
            print(f"Wrote {len(deprecated_df)} rows to {output_path}")

        preview_rows = max(args.rows, 0)
        if preview_rows:
            print(deprecated_df.head(preview_rows).to_string(index=False))

    if args.command == "effects-table":
        effects_df = build_effects_table(data_dir=args.data_dir)
        if args.out:
            output_path = Path(args.out)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            effects_df.to_csv(output_path, index=False)
            print(f"Wrote {len(effects_df)} rows to {output_path}")

        preview_rows = max(args.rows, 0)
        if preview_rows:
            print(effects_df.head(preview_rows).to_string(index=False))

    if args.command == "removed-effects-table":
        removed_effects_df = build_removed_effects_table(data_dir=args.data_dir)
        if args.out:
            output_path = Path(args.out)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            removed_effects_df.to_csv(output_path, index=False)
            print(f"Wrote {len(removed_effects_df)} rows to {output_path}")

        preview_rows = max(args.rows, 0)
        if preview_rows:
            print(removed_effects_df.head(preview_rows).to_string(index=False))

    if args.command == "effect-classes":
        effect_classes_df = build_effect_classes_df(
            data_dir=args.data_dir,
            filtered=not args.unfiltered,
        )
        if args.out:
            output_path = Path(args.out)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            effect_classes_df.to_csv(output_path, index=False)
            print(f"Wrote {len(effect_classes_df)} rows to {output_path}")

        preview_rows = max(args.rows, 0)
        if preview_rows:
            print(effect_classes_df.head(preview_rows).to_string(index=False))

    if args.command == "effect-ranking":
        ranking_df = build_effect_ranking_table(
            data_dir=args.data_dir,
            filtered=args.filtered,
        )
        if args.out:
            output_path = Path(args.out)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            ranking_df.to_csv(output_path, index=False)
            print(f"Wrote {len(ranking_df)} rows to {output_path}")

        preview_rows = max(args.rows, 0)
        if preview_rows:
            print(ranking_df.head(preview_rows).to_string(index=False))


if __name__ == "__main__":
    main()

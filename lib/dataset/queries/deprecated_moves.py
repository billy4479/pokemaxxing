"""Queries for inferring move deprecation across version groups."""

from __future__ import annotations

import pandas as pd

from ..io import DatasetLoader


def _english_move_names(loader: DatasetLoader) -> pd.DataFrame:
    move_names = loader.table("move_names")
    languages = loader.table("languages")

    english_language_ids = languages.loc[languages["identifier"] == "en", "id"]
    english_names = move_names.loc[
        move_names["local_language_id"].isin(english_language_ids),
        ["move_id", "name"],
    ]
    return english_names.drop_duplicates(subset=["move_id"])


def _version_groups(loader: DatasetLoader) -> pd.DataFrame:
    return loader.table("version_groups")[["id", "identifier", "order"]].rename(
        columns={
            "id": "version_group_id",
            "identifier": "version_group",
            "order": "version_group_order",
        }
    )


def infer_deprecated_moves(
    data_dir: str = "csv",
    reference_version_group: str | None = None,
) -> pd.DataFrame:
    """Infer moves absent from the reference latest move-learnset data.

    Notes:
    - This is inferred from `pokemon_moves`, not an explicit deprecation flag.
    - If `reference_version_group` is omitted, the latest version-group order that
      appears in `pokemon_moves` is used.
    """

    loader = DatasetLoader(data_dir=data_dir)

    version_groups = _version_groups(loader)
    moves = loader.table("moves")[["id", "identifier"]].rename(
        columns={"id": "move_id", "identifier": "move_identifier"}
    )
    names = _english_move_names(loader)

    pokemon_moves = loader.table("pokemon_moves")[
        ["move_id", "version_group_id"]
    ].drop_duplicates()
    move_versions = pokemon_moves.merge(
        version_groups, on="version_group_id", how="left"
    )

    if reference_version_group:
        target_rows = version_groups[
            version_groups["version_group"] == reference_version_group
        ]
        if target_rows.empty:
            raise ValueError(f"Unknown version group: {reference_version_group}")
        reference_order = int(target_rows["version_group_order"].iloc[0])
        reference_label = reference_version_group
    else:
        reference_order = int(move_versions["version_group_order"].max())
        reference_label = version_groups.loc[
            version_groups["version_group_order"] == reference_order, "version_group"
        ].iloc[0]

    per_move = move_versions.groupby("move_id").agg(
        first_seen_order=("version_group_order", "min"),
        last_seen_order=("version_group_order", "max"),
        version_group_count=("version_group_id", "nunique"),
    )
    per_move = per_move.reset_index()

    order_lookup = version_groups[
        ["version_group_order", "version_group"]
    ].drop_duplicates()
    per_move = per_move.merge(
        order_lookup.rename(
            columns={
                "version_group_order": "first_seen_order",
                "version_group": "first_seen_version_group",
            }
        ),
        on="first_seen_order",
        how="left",
    )
    per_move = per_move.merge(
        order_lookup.rename(
            columns={
                "version_group_order": "last_seen_order",
                "version_group": "last_seen_version_group",
            }
        ),
        on="last_seen_order",
        how="left",
    )

    deprecated = per_move[per_move["last_seen_order"] < reference_order].copy()
    deprecated["reference_version_group"] = reference_label
    deprecated["reference_order"] = reference_order

    deprecated = deprecated.merge(moves, on="move_id", how="left")
    deprecated = deprecated.merge(names, on="move_id", how="left")

    columns = [
        "move_id",
        "move_identifier",
        "name",
        "last_seen_version_group",
        "last_seen_order",
        "reference_version_group",
        "reference_order",
        "first_seen_version_group",
        "first_seen_order",
        "version_group_count",
    ]
    deprecated = deprecated[columns]

    return deprecated.sort_values(["last_seen_order", "move_identifier"]).reset_index(
        drop=True
    )


def has_explicit_move_deprecation_flag(data_dir: str = "csv") -> bool:
    """Return whether the dataset has an explicit move deprecation field."""

    loader = DatasetLoader(data_dir=data_dir)
    candidate_tables = [
        "moves",
        "move_changelog",
        "pokemon_moves",
        "version_groups",
    ]

    for table_name in candidate_tables:
        columns = loader.table(table_name).columns
        if any("deprecat" in col or "removed" in col for col in columns):
            return True
    return False

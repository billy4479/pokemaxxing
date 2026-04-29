"""Queries related to Pokemon moves."""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from ..io import DatasetLoader
from .deprecated_moves import infer_deprecated_moves


def _normalize_name(value: str) -> str:
    return value.replace("-", "_")


def _english_move_names(loader: DatasetLoader) -> pd.DataFrame:
    move_names = loader.table("move_names")
    languages = loader.table("languages")

    english_language_ids = languages.loc[languages["identifier"] == "en", "id"]
    english_names = move_names.loc[
        move_names["local_language_id"].isin(english_language_ids),
        ["move_id", "name"],
    ]
    return english_names.drop_duplicates(subset=["move_id"])


def _move_effect_labels(loader: DatasetLoader) -> pd.DataFrame:
    """Return effect labels: identifier when available, else effect id."""

    move_effects = loader.table("move_effects")
    if "identifier" in move_effects.columns:
        return move_effects[["id", "identifier"]].rename(
            columns={"id": "effect_id", "identifier": "effect"}
        )

    fallback = move_effects[["id"]].rename(columns={"id": "effect_id"})
    fallback["effect"] = fallback["effect_id"].astype("Int64").astype(str)
    return fallback


def _flags_by_move(loader: DatasetLoader) -> pd.DataFrame:
    flag_map = loader.table("move_flag_map")
    flags = loader.table("move_flags")

    flag_rows = flag_map.merge(
        flags,
        left_on="move_flag_id",
        right_on="id",
        how="left",
    )[["move_id", "identifier"]]

    flag_summary = (
        flag_rows.groupby("move_id")["identifier"]
        .agg(lambda values: "|".join(sorted(set(values))))
        .rename("flags")
        .reset_index()
    )

    flag_matrix = pd.crosstab(flag_rows["move_id"], flag_rows["identifier"]).astype(
        bool
    )
    flag_matrix = flag_matrix.rename(
        columns=lambda value: f"flag_{_normalize_name(value)}"
    )
    flag_matrix = flag_matrix.reset_index()

    return flag_summary.merge(flag_matrix, on="move_id", how="outer")


def _stat_changes_by_move(loader: DatasetLoader) -> pd.DataFrame:
    stat_changes = loader.table("move_meta_stat_changes")
    stats = loader.table("stats")

    change_rows = stat_changes.merge(
        stats,
        left_on="stat_id",
        right_on="id",
        how="left",
    )[["move_id", "identifier", "change"]]
    change_rows = change_rows.rename(columns={"identifier": "stat_identifier"})

    change_rows = change_rows.assign(
        stat_change=change_rows["stat_identifier"]
        + ":"
        + change_rows["change"].map("{:+d}".format)
    )
    summary = (
        change_rows.groupby("move_id")["stat_change"]
        .agg(lambda values: "|".join(sorted(values)))
        .rename("stat_changes")
        .reset_index()
    )

    matrix = change_rows.pivot_table(
        index="move_id",
        columns="stat_identifier",
        values="change",
        aggfunc="sum",
        fill_value=0,
    )
    matrix = matrix.rename(
        columns=lambda value: f"stat_change_{_normalize_name(value)}"
    )
    matrix = matrix.reset_index()

    return summary.merge(matrix, on="move_id", how="outer")


def _rename_columns(df: pd.DataFrame, columns: dict[str, str]) -> pd.DataFrame:
    return df.rename(columns=columns)


def _join_lookup(
    base: pd.DataFrame,
    lookup: pd.DataFrame,
    *,
    left_on: str,
    right_on: str,
    columns: Iterable[str],
) -> pd.DataFrame:
    return base.merge(
        lookup[list(columns)], left_on=left_on, right_on=right_on, how="left"
    )


def _compute_expected_hits(table: pd.DataFrame) -> pd.Series:
    """Compute expected hits for multi-hit moves using simplified rules."""

    expected_hits = pd.Series(1.0, index=table.index)

    min_hits = table["min_hits"]
    max_hits = table["max_hits"]
    has_hit_range = min_hits.notna() & max_hits.notna()

    same_hits = has_hit_range & (min_hits == max_hits)
    expected_hits.loc[same_hits] = min_hits.loc[same_hits]

    standard_two_to_five = has_hit_range & (min_hits == 2) & (max_hits == 5)
    expected_hits.loc[standard_two_to_five] = 3.0

    remaining_ranges = has_hit_range & ~(same_hits | standard_two_to_five)
    expected_hits.loc[remaining_ranges] = (
        min_hits.loc[remaining_ranges] + max_hits.loc[remaining_ranges]
    ) / 2

    return expected_hits


def build_moves_table_full(data_dir: str = "csv") -> pd.DataFrame:
    """Return one battle-focused DataFrame with rich move metadata."""

    loader = DatasetLoader(data_dir=data_dir)

    moves = loader.table("moves")
    names = _english_move_names(loader)
    effects = _move_effect_labels(loader)
    types = loader.table("types")[["id", "identifier"]]
    damage_classes = loader.table("move_damage_classes")[["id", "identifier"]]
    targets = loader.table("move_targets")[["id", "identifier"]]

    move_meta = loader.table("move_meta")
    meta_ailments = loader.table("move_meta_ailments")[["id", "identifier"]]
    meta_categories = loader.table("move_meta_categories")[["id", "identifier"]]

    flags = _flags_by_move(loader)
    stat_changes = _stat_changes_by_move(loader)

    table = moves.rename(columns={"id": "move_id", "identifier": "move_identifier"})
    table = table.merge(names, on="move_id", how="left")
    table = table.merge(effects, on="effect_id", how="left")

    table = _join_lookup(
        table,
        _rename_columns(types, {"id": "type_id", "identifier": "type"}),
        left_on="type_id",
        right_on="type_id",
        columns=["type_id", "type"],
    )
    table = _join_lookup(
        table,
        _rename_columns(
            damage_classes, {"id": "damage_class_id", "identifier": "damage_class"}
        ),
        left_on="damage_class_id",
        right_on="damage_class_id",
        columns=["damage_class_id", "damage_class"],
    )
    table = _join_lookup(
        table,
        _rename_columns(targets, {"id": "target_id", "identifier": "target"}),
        left_on="target_id",
        right_on="target_id",
        columns=["target_id", "target"],
    )

    meta = move_meta.merge(
        _rename_columns(
            meta_ailments, {"id": "meta_ailment_id", "identifier": "meta_ailment"}
        ),
        on="meta_ailment_id",
        how="left",
    )
    meta = meta.merge(
        _rename_columns(
            meta_categories, {"id": "meta_category_id", "identifier": "meta_category"}
        ),
        on="meta_category_id",
        how="left",
    )
    table = table.merge(meta, on="move_id", how="left")

    table = table.merge(flags, on="move_id", how="left")
    table = table.merge(stat_changes, on="move_id", how="left")

    boolean_cols = [col for col in table.columns if col.startswith("flag_")]
    for col in boolean_cols:
        table[col] = table[col].where(table[col].notna(), False).astype(bool)

    stat_change_cols = [col for col in table.columns if col.startswith("stat_change_")]
    for col in stat_change_cols:
        table[col] = table[col].where(table[col].notna(), 0).astype(int)

    text_cols = ["flags", "stat_changes"]
    for col in text_cols:
        table[col] = table[col].fillna("")

    return table.sort_values("move_id").reset_index(drop=True)


def build_moves_table(data_dir: str = "csv") -> pd.DataFrame:
    """Return a simplified unified moves table for analysis.

    This keeps move behavior columns while dropping contest, meta prefix,
    and flag-prefixed columns.
    """

    table = build_moves_table_full(data_dir=data_dir)

    table["effect"] = table["effect"].fillna("")
    table = table[table["pp"].notna()].copy()

    z_moves = DatasetLoader(data_dir=data_dir).table("z_moves_pokemondb")
    z_move_ids = set(z_moves["z_move_identifier"].dropna())

    typed_mask = z_moves["base_move"].isna() & z_moves["pokemon"].isna()
    typed_ids = set(z_moves.loc[typed_mask, "z_move_identifier"].dropna())
    typed_variants = {
        f"{identifier}--{suffix}"
        for identifier in typed_ids
        for suffix in ("physical", "special")
    }
    z_move_ids = z_move_ids.union(typed_variants)

    table = table[~table["move_identifier"].isin(z_move_ids)].copy()

    table["accuracy"] = table["accuracy"].replace(0, 100).fillna(100)
    table["power"] = table["power"].fillna(0)
    table["expected_hits"] = _compute_expected_hits(table)
    table["power"] = (table["power"] * table["expected_hits"]).round().astype(int)

    for col in ["drain", "healing", "crit_rate"]:
        table[col] = table[col].fillna(0)

    table["turns"] = 1.0
    has_turn_range = table["min_turns"].notna() & table["max_turns"].notna()
    table.loc[has_turn_range, "turns"] = (
        table.loc[has_turn_range, "min_turns"] + table.loc[has_turn_range, "max_turns"]
    ) / 2

    chance_cols = [col for col in table.columns if col.endswith("_chance")]
    for col in chance_cols:
        table[col] = table[col].fillna(0)

    drop_columns = [
        "type_id",
        "damage_class_id",
        "effect_id",
        "stat_changes",
        "min_hits",
        "max_hits",
        "min_turns",
        "max_turns",
        "expected_hits",
    ]
    drop_columns.extend(
        col
        for col in table.columns
        if ("contest" in col) or col.startswith("meta_") or col.startswith("flag")
    )
    table = table.drop(columns=[col for col in drop_columns if col in table.columns])

    deprecated_move_ids = set(infer_deprecated_moves(data_dir=data_dir)["move_id"])
    table = table[~table["move_id"].isin(deprecated_move_ids)].copy()

    return table

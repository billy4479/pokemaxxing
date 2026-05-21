"""Queries related to Pokemon abilities."""

from __future__ import annotations

import pandas as pd

from ..io import DatasetLoader


def _english_ability_names(loader: DatasetLoader) -> pd.DataFrame:
    ability_names = loader.table("ability_names")
    languages = loader.table("languages")

    english_language_ids = languages.loc[languages["identifier"] == "en", "id"]
    english_names = ability_names.loc[
        ability_names["local_language_id"].isin(english_language_ids),
        ["ability_id", "name"],
    ]
    return english_names.drop_duplicates(subset=["ability_id"])


def _english_ability_prose(loader: DatasetLoader) -> pd.DataFrame:
    ability_prose = loader.table("ability_prose")
    languages = loader.table("languages")

    english_language_ids = languages.loc[languages["identifier"] == "en", "id"]
    english_prose = ability_prose.loc[
        ability_prose["local_language_id"].isin(english_language_ids),
        ["ability_id", "short_effect", "effect"],
    ]
    return english_prose.drop_duplicates(subset=["ability_id"])


def build_abilities_table_full(data_dir: str = "csv") -> pd.DataFrame:
    """Return one DataFrame with rich ability metadata."""

    loader = DatasetLoader(data_dir=data_dir)

    abilities = loader.table("abilities")
    names = _english_ability_names(loader)
    prose = _english_ability_prose(loader)

    table = abilities.rename(
        columns={"id": "ability_id", "identifier": "ability_identifier"}
    )

    table = table.merge(names, on="ability_id", how="left")
    table = table.merge(prose, on="ability_id", how="left")

    table["name"] = table["name"].fillna("")
    table["short_effect"] = table["short_effect"].fillna("")
    table["effect"] = table["effect"].fillna("")

    return table.sort_values("ability_id").reset_index(drop=True)


def build_abilities_table(data_dir: str = "csv") -> pd.DataFrame:
    """Return a simplified abilities table for analysis.

    Filters to main-series abilities and drops internal ID columns.
    """

    table = build_abilities_table_full(data_dir=data_dir)

    table = table[table["is_main_series"] == 1].copy()

    drop_columns = [
        "ability_id",
        "is_main_series",
    ]
    table = table.drop(columns=drop_columns)

    return table.reset_index(drop=True)

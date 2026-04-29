"""Queries for move effect tables."""

from __future__ import annotations

import json

import pandas as pd

from ..io import DatasetLoader
from ..models import EffectClass, serialize_component
from ..parsers import parse_effects_dataframe


def _build_effects_table_base(data_dir: str = "csv") -> pd.DataFrame:
    """Return the unfiltered unified DataFrame of move effects."""

    loader = DatasetLoader(data_dir=data_dir)

    move_effects = loader.table("move_effects")
    move_effects = move_effects.rename(columns={"id": "effect_id"})

    if "identifier" in move_effects.columns:
        move_effects = move_effects.rename(columns={"identifier": "effect_identifier"})
    else:
        move_effects["effect_identifier"] = (
            move_effects["effect_id"].astype("Int64").astype(str)
        )

    move_effect_prose = loader.table("move_effect_prose")
    languages = loader.table("languages")
    english_language_ids = languages.loc[languages["identifier"] == "en", "id"]

    english_prose = move_effect_prose.loc[
        move_effect_prose["local_language_id"].isin(english_language_ids),
        ["move_effect_id", "short_effect", "effect"],
    ].rename(columns={"move_effect_id": "effect_id"})

    moves = loader.table("moves")[["id", "identifier", "effect_id"]].rename(
        columns={"id": "move_id", "identifier": "move_identifier"}
    )
    effect_usage = moves.groupby("effect_id").agg(
        move_count=("move_id", "nunique"),
        move_ids=("move_id", lambda values: "|".join(map(str, sorted(set(values))))),
    )
    effect_usage = effect_usage.reset_index()

    pokemon_moves = loader.table("pokemon_moves")[
        ["pokemon_id", "move_id"]
    ].drop_duplicates()
    pokemon = loader.table("pokemon")[["id", "species_id"]].rename(
        columns={"id": "pokemon_id"}
    )
    move_to_species = pokemon_moves.merge(pokemon, on="pokemon_id", how="left")

    effect_pokemon_forms_any = (
        moves.merge(pokemon_moves, on="move_id", how="left")
        .groupby("effect_id")
        .agg(pokemon_form_count_any_move=("pokemon_id", "nunique"))
        .reset_index()
    )

    effect_pokemon_species_any = (
        moves.merge(
            move_to_species[["move_id", "species_id"]], on="move_id", how="left"
        )
        .groupby("effect_id")
        .agg(pokemon_species_count_any_move=("species_id", "nunique"))
        .reset_index()
    )

    table = move_effects.merge(english_prose, on="effect_id", how="left")
    table = table.merge(effect_usage, on="effect_id", how="left")
    table = table.merge(effect_pokemon_forms_any, on="effect_id", how="left")
    table = table.merge(effect_pokemon_species_any, on="effect_id", how="left")

    table["short_effect"] = table["short_effect"].fillna("")
    table["effect"] = table["effect"].fillna("")
    table["move_count"] = table["move_count"].fillna(0).astype(int)
    table["move_ids"] = table["move_ids"].fillna("")
    table["pokemon_form_count_any_move"] = (
        table["pokemon_form_count_any_move"].fillna(0).astype(int)
    )
    table["pokemon_species_count_any_move"] = (
        table["pokemon_species_count_any_move"].fillna(0).astype(int)
    )
    table["score_move_count_x_species_any"] = (
        table["move_count"] * table["pokemon_species_count_any_move"]
    )

    return table.sort_values("effect_id").reset_index(drop=True)


def _effects_removal_mask(table: pd.DataFrame) -> pd.Series:
    return table["score_move_count_x_species_any"] < 10


def build_effects_table_full(data_dir: str = "csv") -> pd.DataFrame:
    """Return the full effects table before filtering."""

    return _build_effects_table_base(data_dir=data_dir)


def build_effects_table(data_dir: str = "csv") -> pd.DataFrame:
    """Return filtered effects table for multi-move, described effects."""

    table = _build_effects_table_base(data_dir=data_dir)
    return table[~_effects_removal_mask(table)].reset_index(drop=True)


def build_removed_effects_table(data_dir: str = "csv") -> pd.DataFrame:
    """Return effects excluded by filtering, with removal reasons."""

    table = _build_effects_table_base(data_dir=data_dir).copy()
    mask = _effects_removal_mask(table)
    table["removal_reason"] = ""
    table.loc[mask, "removal_reason"] = "score<10"

    return table[mask].reset_index(drop=True)


def build_effect_classes(
    data_dir: str = "csv", filtered: bool = True
) -> list[EffectClass]:
    """Parse effects into typed effect classes.

    Args:
        data_dir: Directory containing CSV files.
        filtered: If True, parse filtered effects only. If False, parse the
            unfiltered full effects table.
    """

    effects = (
        build_effects_table(data_dir=data_dir)
        if filtered
        else build_effects_table_full(data_dir=data_dir)
    )
    return parse_effects_dataframe(effects)


def build_effect_classes_df(
    data_dir: str = "csv", filtered: bool = True
) -> pd.DataFrame:
    """Return a flattened DataFrame representation of parsed effect classes.

    Args:
        data_dir: Directory containing CSV files.
        filtered: If True, parse filtered effects only. If False, parse the
            unfiltered full effects table.
    """

    classes = build_effect_classes(data_dir=data_dir, filtered=filtered)

    rows: list[dict[str, object]] = []
    for item in classes:
        component_dicts = [
            serialize_component(component) for component in item.components
        ]
        rows.append(
            {
                "effect_id": item.effect_id,
                "effect_key": item.effect_key,
                "component_count": len(item.components),
                "component_types": "|".join(
                    sorted(
                        {
                            f"{component.payload.family}:{component.payload.op}"
                            for component in item.components
                        }
                    )
                ),
                "components_json": json.dumps(component_dicts, sort_keys=True),
                "confidence": item.confidence,
                "raw_short_effect": item.raw_short_effect,
                "raw_effect": item.raw_effect,
            }
        )

    return pd.DataFrame(rows).sort_values("effect_id").reset_index(drop=True)


def build_effect_ranking_table(
    data_dir: str = "csv", filtered: bool = False
) -> pd.DataFrame:
    """Rank effects by move-count times learner-count.

    Score definition:
        score = move_count * pokemon_species_count_any_move

    where `pokemon_species_count_any_move` is the number of unique Pokemon
    species that can learn at least one move with this effect.
    """

    loader = DatasetLoader(data_dir=data_dir)

    effects = (
        build_effects_table(data_dir=data_dir)
        if filtered
        else build_effects_table_full(data_dir=data_dir)
    )

    moves = loader.table("moves")[["id", "effect_id"]].rename(columns={"id": "move_id"})
    pokemon_moves = loader.table("pokemon_moves")[
        ["pokemon_id", "move_id"]
    ].drop_duplicates()
    pokemon = loader.table("pokemon")[["id", "species_id"]].rename(
        columns={"id": "pokemon_id"}
    )

    move_learners = (
        pokemon_moves.groupby("move_id")
        .agg(pokemon_form_count=("pokemon_id", "nunique"))
        .reset_index()
    )

    move_to_species = pokemon_moves.merge(pokemon, on="pokemon_id", how="left")
    move_species_learners = (
        move_to_species.groupby("move_id")
        .agg(pokemon_species_count=("species_id", "nunique"))
        .reset_index()
    )

    move_metrics = moves.merge(move_learners, on="move_id", how="left")
    move_metrics = move_metrics.merge(move_species_learners, on="move_id", how="left")
    move_metrics["pokemon_form_count"] = (
        move_metrics["pokemon_form_count"].fillna(0).astype(int)
    )
    move_metrics["pokemon_species_count"] = (
        move_metrics["pokemon_species_count"].fillna(0).astype(int)
    )

    effect_learner_sums = (
        move_metrics.groupby("effect_id")
        .agg(
            pokemon_form_count_sum=("pokemon_form_count", "sum"),
            pokemon_species_count_sum=("pokemon_species_count", "sum"),
            pokemon_form_count_avg=("pokemon_form_count", "mean"),
            pokemon_species_count_avg=("pokemon_species_count", "mean"),
        )
        .reset_index()
    )

    ranking = effects.copy()
    ranking = ranking.merge(effect_learner_sums, on="effect_id", how="left")

    numeric_cols = [
        "move_count",
        "pokemon_form_count_sum",
        "pokemon_species_count_sum",
        "pokemon_form_count_avg",
        "pokemon_species_count_avg",
        "pokemon_form_count_any_move",
        "pokemon_species_count_any_move",
    ]
    for col in numeric_cols:
        ranking[col] = ranking[col].fillna(0)

    int_cols = [
        "move_count",
        "pokemon_form_count_sum",
        "pokemon_species_count_sum",
        "pokemon_form_count_any_move",
        "pokemon_species_count_any_move",
    ]
    for col in int_cols:
        ranking[col] = ranking[col].astype(int)

    ranking = ranking.sort_values(
        [
            "score_move_count_x_species_any",
            "move_count",
            "pokemon_species_count_any_move",
            "effect_id",
        ],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)

    columns = [
        "effect_id",
        "effect_identifier",
        "short_effect",
        "effect",
        "move_count",
        "pokemon_species_count_any_move",
        "pokemon_form_count_any_move",
        "pokemon_species_count_sum",
        "pokemon_form_count_sum",
        "pokemon_species_count_avg",
        "pokemon_form_count_avg",
        "score_move_count_x_species_any",
    ]
    return ranking[columns]

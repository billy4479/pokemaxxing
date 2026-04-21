"""Dataset query library for Pokemon CSV files."""

from .queries.deprecated_moves import (
    has_explicit_move_deprecation_flag,
    infer_deprecated_moves,
)
from .queries.effects import (
    build_effect_classes,
    build_effect_classes_df,
    build_effect_ranking_table,
    build_effects_table,
    build_effects_table_full,
    build_removed_effects_table,
)
from .queries.moves import build_moves_table

__all__ = [
    "build_moves_table",
    "build_effects_table",
    "build_effects_table_full",
    "build_removed_effects_table",
    "build_effect_classes",
    "build_effect_classes_df",
    "build_effect_ranking_table",
    "infer_deprecated_moves",
    "has_explicit_move_deprecation_flag",
]

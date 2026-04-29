"""Parsers that convert prose to typed effect classes."""

from .effects_parser import parse_effect_row, parse_effects_dataframe

__all__ = ["parse_effect_row", "parse_effects_dataframe"]

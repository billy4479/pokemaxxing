import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")

with app.setup:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    import marimo as mo

    from lib.dataset.queries import (
        build_moves_table,
    )

    from lib.dataset.queries.effects import (
        build_effect_classes,
        build_effects_table,
    )


@app.cell
def _():
    df_moves = build_moves_table()
    df_moves
    return (df_moves,)


@app.cell
def _(df_moves):
    df_moves.describe()
    return


@app.cell
def _():
    df_effects = build_effects_table()
    df_effects
    return


@app.cell
def _():
    effects = build_effect_classes()
    return (effects,)


@app.cell
def _(effects):
    effects
    return


@app.cell
def _(effects):
    markdown = ""

    for effect in effects:
        markdown += effect.to_markdown() + "\n"*4

    mo.md(markdown)
    return


if __name__ == "__main__":
    app.run()

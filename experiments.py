import marimo

__generated_with = "0.21.1"
app = marimo.App(
    width="full",
    css_file="/home/billy/Downloads/catppuccin-latte-frappe.css",
)

with app.setup:
    from dataclasses import dataclass
    from enum import Enum
    from functools import lru_cache
    from pathlib import Path
    from typing import Any

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import showdown_wrapper as sdw
    import torch
    from deap import base, creator, tools
    from scipy.spatial.distance import cdist
    from scipy.special import softmax
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import (
        pairwise_distances,
        precision_recall_fscore_support,
        silhouette_score,
    )
    from sklearn.preprocessing import StandardScaler
    from torch import nn
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, TensorDataset
    from tqdm import tqdm
    from umap import UMAP

    from lib.dataset.queries import build_moves_table


@app.cell
def _():
    plt.rcParams["figure.figsize"] = (12, 6)
    plt.rcParams["font.size"] = 16
    plt.rcParams["axes.titlesize"] = 18
    plt.rcParams["axes.labelsize"] = 16
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12
    plt.rcParams["legend.fontsize"] = 12
    return


@app.cell
def _():
    seed = 42

    np.random.seed(seed)
    rng = np.random.default_rng(seed + 1)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device, rng, seed


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # PokéMaxxing - What is the ideal Pokémon?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Pokémon is a very famous game where two players let their pokémon fight in battles.
    The dynamics of the battles are quite complex and each player has to choose their strategy carefully before going into a battle.

    In this project I try to answer the following questions:
    > If it existed a pokémon which could learn all moves, we could choose its stats, its type and its ability, what would the best choice so that that pokémon could beat as many adversaries as possible?

    I intend to answer this question by training an evolutionary model, its genotype will be made of
    - 4 moves within the existing ones (according to the rules, each pokémons have 4 moves available in each battle);
    - 6 base statistics so that their sum is below a certain threshold:
        - Hit Points (HP)
        - Attach (ATK)
        - Defense (DEF)
        - Special Attach (SPA)
        - Special Defense (SPD)
        - Speed (SPE)
    - 2 types within the existing ones (which can be the same).

    Most importantly, to evaluate the fitness of each agent they will need to fight against other pokémons, so I will evolve a small MLP which will choose what to do in the fight (I will refer to it as the "battle MLP").
    While the topology of the battle MLP will be fixed, the weights will evolve alongside the other genes (so no gradient descent here).
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Technical notes

    Luckily, since pokémon is a very famous franchise, a lot of great tools already exist.

    - I used the [PokéAPI](https://github.com/PokeAPI/pokeapi/) dataset as the main dataset for this project. This dataset is licensed under the [BSD-3-Clause License](https://github.com/PokeAPI/pokeapi/blob/master/LICENSE.md).
    - The battle logic is implemented by the awesome [Pokémon Showdown](https://github.com/smogon/pokemon-showdown) library, licensed under the [MIT License](https://github.com/smogon/pokemon-showdown).

    Pokémon Showdown is the most complete implementation of Pokémon battles, however it is written in TypeScript.
    To use it in this project I wrote a "worker" interface to the library with the logic I need which can be controlled by piping commands through stdio.
    Then I wrote a Python program which manages a pool of these NodeJS worker processes so that the decision making on which move to play is delegated to the Python code.
    In this way I can execute multiple battles in parallel while still controlling everything thorugh Python.
    The code for the worker and the wrapper lives in [this repository](https://github.com/billy4479/pokemon-showdown-wrapper).

    While most of the code for this project is written by me, I was assisted by AI, expecially in the prototyping phase and in the parts which were not completely relevant to the evolutionary model part of the project.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Limitations

    The game of Pokémon is very complex and some parts are very much RNG based. Because of time and hardware constraints I added some rules which simplify the battle, not all of these rules are very realistic but otherwise the MLP would have had to be much bigger and the training would have been much longer.

    To be precise, the battle rules that the Showdown runner is using are the following:
    ```ts
    export const customRules = [
        "Picked Team Size = 1",
        "Max Team Size = 1",
        "Min Team Size = 1",

        "Terastal Clause",
        "Dynamax Clause",
        "Z-Move Clause",
        "CFZ Clause",

        "-Dragon Ascent",
        "-pokemontag:allitems",

        "OHKO Clause",
        "Evasion Clause",
        // "Accuracy Moves Clause",
        "Sleep Moves Clause",
        "Freeze Clause Mod",
        "Moody Clause",
        "Swagger Clause",

        "Endless Battle Clause",
        "Exact HP Mod",

        "-All Abilities",
        "+No Ability",
    ].join(",");
    ```

    These limit the amout of moves each pokomon can learn and makes the battle simpler.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Encoding moves - AutoEncoder

    Our first task is kind of off topic: we need to find a way to tell the battle MLP which moves our pokémon chose to learn.
    This is quite a hard task: there exist more than 900 moves in the game and each one of them is increadibly rich in information content:
    they have various many numerical statistics and a lot of categorical features such as the type or the effect.

    In particular the effect is problematic as they are so complicated that no dataset even attempts to describe what each effect does numerically, preferring a text description instead.

    In this section I explore how to encode moves in a lower dimensional space trying to preserve meaning: moves which gets used in similar situations should be close together in the resulting embedding, so that the resulting compressed move can be fed to the MLP without blowing up the number of weights.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Preprocessing

    I started by applying some preprocessing to the moves dataset from PokéAPI, simplifying some mechanics
    and removing Z-Moves as they add a non-trivial amount of complexity which is not really in the scope for this project.

    After many attemps I chose to drop the `effect` column: this was just a id which could have been used to look up the effect description (in english).
    There are more than 300 unique effects in the game so adding this huge one-hot encoded vector without giving any meaning to it turned out to be counterproductive:
    training was focussing too much on reconstructing the effect vector correctly rather than focussing on the tactical meaning of the move.
    """)
    return


@app.cell
def _():
    moves_df = build_moves_table().drop(columns=["effect"])
    moves_df["move_identifier"] = moves_df["move_identifier"].map(
        lambda x: x.replace("-", "")
    )

    _sd_moves = set(sdw.list_allowed_moves())
    _pokeapi_moves = set(moves_df["move_identifier"])
    _intersection = list(_sd_moves.intersection(_pokeapi_moves))

    moves_df = moves_df[moves_df["move_identifier"].isin(_intersection)]
    moves_df
    return (moves_df,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    I started by reprenting this table in a way in which it can be fed to an autoencoder more easily, in particular I one-hot encoded the categorical variables.
    """)
    return


@app.cell
def _(moves_df):
    metadata_cols = ["move_id", "move_identifier"]
    categorical_cols = (
        moves_df.select_dtypes(include=["object"])
        .columns.drop(["move_identifier"])
        .tolist()
    )

    flag_cols = [col for col in moves_df.columns if col.startswith("flag_")]
    stat_change_cols = [
        col for col in moves_df.columns if col.startswith("stat_change_")
    ]

    numeric_cols = [
        col
        for col in moves_df.columns
        if col not in metadata_cols + categorical_cols + flag_cols + stat_change_cols
        and pd.api.types.is_numeric_dtype(moves_df[col])
    ]

    {
        "metadata": metadata_cols,
        "categorical": categorical_cols,
        "flags": flag_cols,
        "numeric": numeric_cols,
        "stat_change": stat_change_cols,
    }
    return (
        categorical_cols,
        flag_cols,
        metadata_cols,
        numeric_cols,
        stat_change_cols,
    )


@app.cell
def _(moves_df, numeric_cols):
    numeric_df = moves_df[numeric_cols].astype(np.float32)
    numeric_mean = numeric_df.mean(axis=0)
    numeric_std = numeric_df.std(axis=0, ddof=0).replace(0.0, 1.0)
    numeric_scaled_df = ((numeric_df - numeric_mean) / numeric_std).astype(np.float32)
    return (numeric_scaled_df,)


@app.cell
def _(categorical_cols, moves_df):
    categorical_df = pd.get_dummies(
        moves_df[categorical_cols].astype("category"),
        prefix=categorical_cols,
        dtype=np.float32,
    )
    return (categorical_df,)


@app.cell
def _(flag_cols, moves_df):
    flags_df = moves_df[flag_cols].astype(np.float32)
    return (flags_df,)


@app.cell
def _(moves_df, stat_change_cols):
    stat_change_df = moves_df[stat_change_cols].astype(np.float32)
    return (stat_change_df,)


@app.cell
def _(categorical_df, flags_df, numeric_scaled_df, stat_change_df):
    feature_df = pd.concat(
        [
            numeric_scaled_df,
            stat_change_df,
            categorical_df,
            flags_df,
        ],
        axis=1,
    )

    feature_df.shape
    return (feature_df,)


@app.cell
def _(
    categorical_cols,
    categorical_df,
    feature_df,
    flags_df,
    numeric_scaled_df,
    stat_change_df,
):
    feature_blocks = {
        "numeric": numeric_scaled_df.columns.tolist(),
        "stat_change": stat_change_df.columns.tolist(),
        "categorical": categorical_df.columns.tolist(),
        "flags": flags_df.columns.tolist(),
    }

    feature_groups = {}
    start = 0
    for group_name, columns in feature_blocks.items():
        end = start + len(columns)
        feature_groups[group_name] = {
            "columns": columns,
            "slice": slice(start, end),
            "size": len(columns),
        }
        start = end

    assert start == feature_df.shape[1]

    column_to_index = {column: index for index, column in enumerate(feature_df.columns)}

    categorical_feature_groups = {}
    for categorical_col in categorical_cols:
        dummy_columns = [
            column
            for column in categorical_df.columns
            if column.startswith(f"{categorical_col}_")
        ]

        categorical_feature_groups[categorical_col] = {
            "columns": dummy_columns,
            "indices": [column_to_index[column] for column in dummy_columns],
            "size": len(dummy_columns),
        }

    group_loss_weights = {
        "numeric": 1.0,
        "stat_change": 1.0,
        "categorical": 1.0,
        "flags": 2.5,
    }

    {
        "feature_groups": {
            group_name: group["size"] for group_name, group in feature_groups.items()
        },
        "categorical_feature_groups": {
            group_name: group["size"]
            for group_name, group in categorical_feature_groups.items()
        },
        "group_loss_weights": group_loss_weights,
    }
    return categorical_feature_groups, feature_groups, group_loss_weights


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The model

    The actual autoencoder is quite small, just two layers with not many dimensions and, in particular, a very small latent space.

    This choice was made so that the battle MLP
    (which will be also very tiny since I won't be able to train it with gradient descent)
    doesn't need much "effort" to decode what a move actually does.

    I'm not interesed in the model generalizing, it is fine if it overfits since the moves it will have to encode are always the same,
    however I decided to include a `Dropout`: this will make the decode "dumber" which should match the battle MLP which will indeed will also be not very smart.
    """)
    return


@app.cell
def _(categorical_feature_groups, feature_groups, group_loss_weights):
    class MoveAutoencoder(nn.Module):
        def __init__(self, input_dim, latent_dim, hidden_dims=(256, 128)):
            super().__init__()

            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dims[0]),
                nn.SiLU(),
                nn.Linear(hidden_dims[0], hidden_dims[1]),
                nn.SiLU(),
                nn.Linear(hidden_dims[1], latent_dim),
                nn.LayerNorm(latent_dim),
            )

            self.decoder = nn.Sequential(
                nn.Dropout(p=0.1),
                nn.Linear(latent_dim, hidden_dims[1]),
                nn.SiLU(),
                nn.Linear(hidden_dims[1], hidden_dims[0]),
                nn.SiLU(),
                nn.Linear(hidden_dims[0], input_dim),
            )

        def encode(self, _inputs):
            return self.encoder(_inputs)

        def decode(self, _latent):
            return self.decoder(_latent)

        def forward(self, _inputs):
            _latent = self.encode(_inputs)
            _reconstruction_logits = self.decode(_latent)
            return _reconstruction_logits, _latent

    def _zero_like_loss(inputs):
        return inputs.sum() * 0.0

    def _mse_group_loss(reconstruction_logits, inputs, group_name):
        group_slice = feature_groups[group_name]["slice"]

        if feature_groups[group_name]["size"] == 0:
            return _zero_like_loss(inputs)

        return F.mse_loss(
            reconstruction_logits[:, group_slice],
            inputs[:, group_slice],
            reduction="mean",
        )

    def _bce_group_loss(reconstruction_logits, inputs, group_name):
        group_slice = feature_groups[group_name]["slice"]

        if feature_groups[group_name]["size"] == 0:
            return _zero_like_loss(inputs)

        return F.binary_cross_entropy_with_logits(
            reconstruction_logits[:, group_slice],
            inputs[:, group_slice],
            reduction="mean",
        )

    def _categorical_grouped_cross_entropy_loss(reconstruction_logits, inputs):
        categorical_losses = []

        for categorical_group in categorical_feature_groups.values():
            indices = categorical_group["indices"]

            if len(indices) <= 1:
                continue

            group_logits = reconstruction_logits[:, indices]
            target_classes = inputs[:, indices].argmax(dim=1)

            categorical_losses.append(
                F.cross_entropy(
                    group_logits,
                    target_classes,
                    reduction="mean",
                )
            )

        if len(categorical_losses) == 0:
            return _zero_like_loss(inputs)

        return torch.stack(categorical_losses).mean()

    def ae_loss(
        reconstruction_logits,
        inputs,
        weights=None,
    ):
        if weights is None:
            weights = group_loss_weights

        recon_numeric = _mse_group_loss(
            reconstruction_logits,
            inputs,
            "numeric",
        )

        recon_stat_change = _mse_group_loss(
            reconstruction_logits,
            inputs,
            "stat_change",
        )

        recon_categorical = _categorical_grouped_cross_entropy_loss(
            reconstruction_logits,
            inputs,
        )

        recon_flags = _bce_group_loss(
            reconstruction_logits,
            inputs,
            "flags",
        )

        loss_parts = {
            "recon_numeric": recon_numeric,
            "recon_stat_change": recon_stat_change,
            "recon_categorical": recon_categorical,
            "recon_flags": recon_flags,
        }

        total = (
            weights["numeric"] * recon_numeric
            + weights["stat_change"] * recon_stat_change
            + weights["categorical"] * recon_categorical
            + weights["flags"] * recon_flags
        )

        return total, loss_parts

    return MoveAutoencoder, ae_loss


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Training

    Since the move table is finite and the autoencoder is used as a preprocessing step,
    I evaluate reconstruction quality on the full dataset rather than using a train/validation split.
    """)
    return


@app.cell
def _(MoveAutoencoder, ae_loss, device):
    def train(
        dataset: pd.DataFrame,
        latent_dim: int,
        epochs: int,
        loss_weights,
        learning_rate: float = 1e-4,
        use_cache=False,
    ):
        batch_size = 128
        cache_path = "move_autoencoder.pt"

        x = torch.tensor(dataset.to_numpy(), dtype=torch.float32)
        tensor_dataset = TensorDataset(x)
        dataloader = DataLoader(
            tensor_dataset,
            batch_size=batch_size,
            shuffle=True,
        )

        input_dim = x.shape[1]
        dataset_size = len(tensor_dataset)

        model = MoveAutoencoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
        ).to(device)

        using_cache = False

        if use_cache and Path(cache_path).exists():
            model.load_state_dict(torch.load(cache_path, weights_only=True))
            using_cache = True
        else:
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=5e-3,
            )

            history_rows = []
            epoch_progress = tqdm(
                range(1, epochs + 1),
                desc="Training AE",
                unit="epoch",
            )

            for epoch in epoch_progress:
                model.train()

                epoch_totals = {
                    "loss": 0.0,
                    "recon_numeric": 0.0,
                    "recon_stat_change": 0.0,
                    "recon_categorical": 0.0,
                    "recon_flags": 0.0,
                }

                for (batch,) in dataloader:
                    batch = batch.to(device)

                    optimizer.zero_grad()

                    reconstruction_logits_batch, latent_batch = model(batch)

                    loss, loss_parts = ae_loss(
                        reconstruction_logits_batch,
                        batch,
                        weights=loss_weights,
                    )

                    loss.backward()
                    optimizer.step()

                    batch_count = batch.shape[0]

                    epoch_totals["loss"] += loss.item() * batch_count
                    for loss_name, loss_value in loss_parts.items():
                        epoch_totals[loss_name] += loss_value.item() * batch_count

                history_row = {
                    "epoch": epoch,
                    **{
                        key: value / dataset_size for key, value in epoch_totals.items()
                    },
                }

                history_rows.append(history_row)

                epoch_progress.set_postfix(
                    loss=f"{history_row['loss']:.4f}",
                    num=f"{history_row['recon_numeric']:.4f}",
                    stat=f"{history_row['recon_stat_change']:.4f}",
                    cat=f"{history_row['recon_categorical']:.4f}",
                    flags=f"{history_row['recon_flags']:.4f}",
                )

            torch.save(model.state_dict(), cache_path)

        model.eval()
        with torch.no_grad():
            x_device = x.to(device)
            latent_embedding = model.encode(x_device)
            reconstruction_logits_eval = model.decode(latent_embedding)

        embedding_matrix = latent_embedding.detach().cpu().numpy()
        reconstruction_logits_matrix = reconstruction_logits_eval.detach().cpu().numpy()

        return (
            model,
            embedding_matrix,
            reconstruction_logits_matrix,
            pd.DataFrame(history_rows) if not using_cache else None,
        )

    return (train,)


@app.cell
def _(feature_df, group_loss_weights, train):
    (
        move_autoencoder,
        embedding_matrix,
        reconstruction_logits_matrix,
        training_history,
    ) = train(
        feature_df,
        latent_dim=8,
        epochs=2**10,
        loss_weights=group_loss_weights,
        use_cache=True,
    )

    if training_history is None:
        print("Weights loaded from cache.")
    return embedding_matrix, reconstruction_logits_matrix, training_history


@app.cell
def _(device, training_history):
    def get_training_summary():
        if training_history is None:
            return None

        final_training_row = training_history.iloc[-1]

        return {
            "device": str(device),
            "epochs": int(training_history["epoch"].max()),
            "final_loss": float(final_training_row["loss"]),
            "best_loss": float(training_history["loss"].min()),
            "final_recon_numeric": float(final_training_row["recon_numeric"]),
            "final_recon_stat_change": float(final_training_row["recon_stat_change"]),
            "final_recon_categorical": float(final_training_row["recon_categorical"]),
            "final_recon_flags": float(final_training_row["recon_flags"]),
        }

    get_training_summary()
    return


@app.cell
def _(training_history):
    def plot_training():
        if training_history is None:
            return

        fig, ax = plt.subplots()

        ax.plot(
            training_history["epoch"],
            training_history["loss"],
            label="total",
            linewidth=2.5,
        )

        ax.plot(
            training_history["epoch"],
            training_history["recon_numeric"],
            label="numeric",
            linewidth=1.5,
        )

        ax.plot(
            training_history["epoch"],
            training_history["recon_stat_change"],
            label="stat_change",
            linewidth=1.5,
        )

        ax.plot(
            training_history["epoch"],
            training_history["recon_categorical"],
            label="categorical",
            linewidth=1.5,
        )

        ax.plot(
            training_history["epoch"],
            training_history["recon_flags"],
            label="flags",
            linewidth=1.5,
        )

        fig.suptitle("Training Progress")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.25)
        ax.legend()

        fig.tight_layout()
        return fig

    plot_training()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### On the choice of hyperparameters

    - The learning rate was made low and the number of epochs high as we don't care if the autoencoder overfits or if the training takes longer, we just want the best result with the least possible number of dimensions in the latent space.
    - Flags used to have a worse reconstruction than other parameters, so I bumped their loss weight so that the model would optimize for them more aggressively.
    - I experimented a lot with the dimension of the latent space. It seems like a higher number of dimensions may help when the number of epochs is low, however with enough training the reconstruction is fine even with as little as 8 dimensions. Lower than 8 we start losing some structure, but I will comment on this again later in the clustering section.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Reconstruction evaluation

    These metrics should be read as compression diagnostics:
    they tell us how much information is preserved by the latent representation.

    Different feature families need different metrics:

    - numeric values and raw stat-change magnitudes use regression metrics;
    - true categorical variables use grouped multiclass accuracy;
    - flags use binary classification metrics;
    - stat-change-by-target flags use binary metrics plus exact-match accuracy.

    Note that we are not just interested to how well the data compresses and gets reconstructed, but also if the latent space carries geometrical information about the tactical meaning of the move.
    """)
    return


@app.cell
def _(
    categorical_feature_groups,
    feature_df,
    feature_groups,
    reconstruction_logits_matrix,
):
    reconstruction_logits_df = pd.DataFrame(
        reconstruction_logits_matrix,
        columns=feature_df.columns,
        index=feature_df.index,
    )

    reconstruction_probability_df = pd.DataFrame(
        1.0 / (1.0 + np.exp(-reconstruction_logits_matrix)),
        columns=feature_df.columns,
        index=feature_df.index,
    )

    def _safe_binary_metrics(y_true, y_pred):
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true.reshape(-1),
            y_pred.reshape(-1),
            average="binary",
            zero_division=0,
        )

        accuracy = float((y_true == y_pred).mean())

        return {
            "accuracy": accuracy,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(y_true.sum()),
        }

    def evaluate_numeric_group(group_name):
        group_slice = feature_groups[group_name]["slice"]
        group_columns = feature_groups[group_name]["columns"]

        y_true = feature_df.iloc[:, group_slice].to_numpy()
        y_pred = reconstruction_logits_df.iloc[:, group_slice].to_numpy()

        errors = y_pred - y_true

        summary = {
            "mae": float(np.abs(errors).mean()),
            "rmse": float(np.sqrt((errors**2).mean())),
        }

        per_column = pd.DataFrame(
            {
                "column": group_columns,
                "mae": np.abs(errors).mean(axis=0),
                "rmse": np.sqrt((errors**2).mean(axis=0)),
            }
        ).sort_values("mae", ascending=False)

        return summary, per_column

    def evaluate_categorical_groups():
        rows = []

        for (
            categorical_name,
            categorical_group,
        ) in categorical_feature_groups.items():
            indices = categorical_group["indices"]

            if len(indices) <= 1:
                continue

            true_class = feature_df.iloc[:, indices].to_numpy().argmax(axis=1)
            pred_class = (
                reconstruction_logits_df.iloc[:, indices].to_numpy().argmax(axis=1)
            )

            rows.append(
                {
                    "categorical": categorical_name,
                    "n_classes": len(indices),
                    "accuracy": float((true_class == pred_class).mean()),
                }
            )

        per_categorical = pd.DataFrame(rows).sort_values(
            "accuracy",
            ascending=True,
        )

        summary = {
            "mean_accuracy": float(per_categorical["accuracy"].mean()),
            "min_accuracy": float(per_categorical["accuracy"].min()),
        }

        return summary, per_categorical

    def evaluate_binary_group(group_name):
        group_slice = feature_groups[group_name]["slice"]
        group_columns = feature_groups[group_name]["columns"]

        y_true = feature_df.iloc[:, group_slice].to_numpy().astype(bool)
        y_pred = reconstruction_probability_df.iloc[:, group_slice].to_numpy() >= 0.5

        micro = _safe_binary_metrics(y_true, y_pred)

        rows = []
        for column_index, column in enumerate(group_columns):
            column_metrics = _safe_binary_metrics(
                y_true[:, column_index],
                y_pred[:, column_index],
            )
            rows.append(
                {
                    "column": column,
                    **column_metrics,
                }
            )

        per_column = pd.DataFrame(rows).sort_values("f1", ascending=True)

        summary = {
            "accuracy": micro["accuracy"],
            "precision": micro["precision"],
            "recall": micro["recall"],
            "micro_f1": micro["f1"],
            "macro_f1": float(per_column["f1"].mean()),
        }

        return summary, per_column

    return (
        evaluate_binary_group,
        evaluate_categorical_groups,
        evaluate_numeric_group,
    )


@app.cell
def _(
    evaluate_binary_group,
    evaluate_categorical_groups,
    evaluate_numeric_group,
):
    numeric_summary, numeric_eval_by_column = evaluate_numeric_group("numeric")
    stat_change_summary, stat_change_eval_by_column = evaluate_numeric_group(
        "stat_change"
    )

    categorical_summary, categorical_eval_by_group = evaluate_categorical_groups()

    flags_summary, flags_eval_by_column = evaluate_binary_group("flags")

    reconstruction_summary = pd.DataFrame(
        [
            {
                "group": "numeric",
                "metric": "mae_scaled",
                "value": numeric_summary["mae"],
            },
            {
                "group": "numeric",
                "metric": "rmse_scaled",
                "value": numeric_summary["rmse"],
            },
            {
                "group": "stat_change",
                "metric": "mae",
                "value": stat_change_summary["mae"],
            },
            {
                "group": "stat_change",
                "metric": "rmse",
                "value": stat_change_summary["rmse"],
            },
            {
                "group": "categorical",
                "metric": "mean_accuracy",
                "value": categorical_summary["mean_accuracy"],
            },
            {
                "group": "flags",
                "metric": "macro_f1",
                "value": flags_summary["macro_f1"],
            },
            {
                "group": "flags",
                "metric": "micro_f1",
                "value": flags_summary["micro_f1"],
            },
        ]
    )

    reconstruction_summary
    return (
        categorical_eval_by_group,
        flags_eval_by_column,
        numeric_eval_by_column,
        reconstruction_summary,
        stat_change_eval_by_column,
    )


@app.cell
def _(reconstruction_summary):
    _plot_df = reconstruction_summary.copy()

    _direction_map = {
        "mae_scaled": "lower is better",
        "rmse_scaled": "lower is better",
        "mae": "lower is better",
        "rmse": "lower is better",
        "mean_accuracy": "higher is better",
        "macro_f1": "higher is better",
        "micro_f1": "higher is better",
    }

    _groups = _plot_df["group"].unique()

    _fig, _axes = plt.subplots(
        1, len(_groups), figsize=(5 * len(_groups), 4), sharey=False
    )

    for _ax, _group in zip(_axes, _groups):
        _group_df = _plot_df[_plot_df["group"] == _group]
        _labels = _group_df["metric"].values
        _values = _group_df["value"].values
        _direction = _direction_map[_labels[0]]

        _bars = _ax.bar(_labels, _values)
        _ax.set_title(_group)
        _ax.tick_params(axis="x", rotation=20)
        _ax.grid(axis="y", alpha=0.25)

        _ymin, _ymax = _ax.get_ylim()
        _y_range = _ymax - _ymin
        _ax.set_ylim(_ymin - 0.05 * _y_range, _ymax + 0.05 * _y_range)

        for _bar, _val in zip(_bars, _values):
            _ax.text(
                _bar.get_x() + _bar.get_width() / 2,
                _bar.get_height(),
                f"{_val:.4f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    _fig.suptitle("Reconstruction Evaluation Summary")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(numeric_eval_by_column):
    numeric_eval_by_column
    return


@app.cell
def _(stat_change_eval_by_column):
    stat_change_eval_by_column
    return


@app.cell
def _(categorical_eval_by_group):
    categorical_eval_by_group
    return


@app.cell
def _(categorical_eval_by_group):
    _plot_df = categorical_eval_by_group.sort_values(
        "accuracy",
        ascending=True,
    )

    _fig, _ax = plt.subplots(figsize=(10, 5))

    _ax.barh(
        _plot_df["categorical"],
        _plot_df["accuracy"],
    )

    _ax.set_title("Categorical Reconstruction Accuracy")
    _ax.set_xlabel("Accuracy")
    _ax.set_xlim(0.0, 1.0)
    _ax.grid(axis="x", alpha=0.25)

    _fig.tight_layout()
    _fig
    return


@app.cell
def _(flags_eval_by_column):
    flags_eval_by_column
    return


@app.cell
def _(flags_eval_by_column):
    _plot_df = flags_eval_by_column.sort_values(
        "f1",
        ascending=True,
    )

    _fig, _ax = plt.subplots(figsize=(10, 6))

    _ax.barh(
        _plot_df["column"],
        _plot_df["f1"],
    )

    _ax.set_title("Reconstructed Flags F1")
    _ax.set_xlabel("F1")
    _ax.set_xlim(0.0, 1.0)
    _ax.grid(axis="x", alpha=0.25)

    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Understanding the structure of the latent space

    Even if the reconstruction accuracy is good it doesn't mean that the latent space is in a good shape for the battle MLP.

    I tried plotting some UMAPs and PCAs and some clustering algorithms.
    """)
    return


@app.cell
def _(seed):
    def compute_umap_pca(embedding_matrix, n_components=2):
        umap = UMAP(
            random_state=seed,
            n_neighbors=10,
            min_dist=0.75,
            metric="euclidean",
            n_jobs=1,
            n_components=n_components,
        ).fit_transform(embedding_matrix)
        pca = PCA(n_components=n_components).fit_transform(embedding_matrix)
        return (umap.T, pca.T)

    return (compute_umap_pca,)


@app.function
def plot_umap_pca_2d(umap, pca, labels, get_mask, title, legend_title="", legend=True):
    fig, (ax1, ax2) = plt.subplots(1, 2)

    fig.suptitle(title)
    fig.tight_layout()

    cmap = plt.get_cmap("tab20")
    colors_per_label = {label: cmap(i % cmap.N) for i, label in enumerate(labels)}

    def scatter(ax, data):
        ax.grid(alpha=0.25)

        for label in labels:
            mask = get_mask(label)
            ax.scatter(
                data[0, mask],
                data[1, mask],
                s=10,
                alpha=0.7,
                color=colors_per_label[label],
                label=label,
            )

    ax1.set_title("UMAP")
    scatter(ax1, umap)

    ax2.set_title("PCA")
    scatter(ax2, pca)

    if legend:
        ax2.legend(title=legend_title, ncols=2, loc="lower right")

    return fig


@app.cell
def _(compute_umap_pca, embedding_matrix):
    umap, pca = compute_umap_pca(embedding_matrix)
    return pca, umap


@app.cell
def _(moves_df, pca, umap):
    types_unique = moves_df["type"].unique().tolist()
    types_unique.sort()

    plot_umap_pca_2d(
        umap,
        pca,
        types_unique,
        lambda type: moves_df["type"] == type,
        "2D Embeddings colored by type",
        "type",
    )
    return (types_unique,)


@app.cell
def _(seed):
    def best_kmeans_silhouette(data, max_k):
        data = StandardScaler().fit_transform(data)
        best_score = -float("inf")
        best_labels = None
        best_k = 0
        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=seed)
            labels = kmeans.fit_predict(data)
            score = silhouette_score(data, labels)
            if score > best_score:
                best_score = score
                best_labels = labels
                best_k = k

                print(f"k: {k} -> {best_score}")

        return best_labels, best_k

    return (best_kmeans_silhouette,)


@app.cell
def _(best_kmeans_silhouette, embedding_matrix, pca, umap):
    kmeans_labels, k = best_kmeans_silhouette(embedding_matrix, max_k=64)
    plot_umap_pca_2d(
        umap,
        pca,
        list(range(k)),
        lambda label: kmeans_labels == label,
        f"2D Embedding colored by KMeans - {k} clusters",
        legend=False,
    )
    return k, kmeans_labels


@app.cell
def _(compute_umap_pca, embedding_matrix):
    umap_3, pca_3 = compute_umap_pca(embedding_matrix, n_components=3)
    return (umap_3,)


@app.function
def plot_embedding_3d(embedding_3, labels, title: str):
    import plotly.graph_objects as go

    df = pd.DataFrame(
        {
            "UMAP1": embedding_3[0],
            "UMAP2": embedding_3[1],
            "UMAP3": embedding_3[2],
            "cluster": labels,
        }
    )

    fig = go.Figure()

    # Normal clusters
    clustered = df[df.cluster != -1]

    for c in sorted(clustered.cluster.unique()):
        sub = clustered[clustered.cluster == c]

        fig.add_trace(
            go.Scatter3d(
                x=sub.UMAP1,
                y=sub.UMAP2,
                z=sub.UMAP3,
                mode="markers",
                name=f"cluster {c}",
                marker=dict(size=2, opacity=0.8),
            )
        )

    # Noise / outliers
    noise = df[df.cluster == -1]

    fig.add_trace(
        go.Scatter3d(
            x=noise.UMAP1,
            y=noise.UMAP2,
            z=noise.UMAP3,
            mode="markers",
            name="noise",
            marker=dict(
                size=1,
                color="lightgray",
                opacity=0.7,
            ),
        )
    )

    fig.update_layout(
        title=title,
        width=1000,
        height=800,
    )

    fig.show()


@app.cell
def _(k, kmeans_labels, umap_3):
    plot_embedding_3d(umap_3, kmeans_labels, f"3D UMAP colored by KMeans - k={k}")
    return


@app.cell
def _(embedding_matrix, umap_3):
    from sklearn.cluster import OPTICS

    optics_labels = OPTICS().fit_predict(embedding_matrix)
    n_clusters_optics = len(np.unique(optics_labels)) - 1
    plot_embedding_3d(
        umap_3,
        optics_labels,
        f"3D UMAP colored by OPTICS - n_clusters={n_clusters_optics}",
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    It's interesting to see that moves did not cluster by type, but, by inspecting some of the clusters, they actually seem clustered by tactical meaning.
    Not all clusters are equally good: some are a bit vague or contain moves which cannot be discriminated further without reading the effect (which my AE is not doing).

    I noted that KMeans tends to like creating a LOT of clusters, by allowing a higher number of maximum clusters I found that setting k=100 or more gives a higher silohuette score, but then clusters don't give much information about the moves themselves and tend to be too specialized.
    A more correct number of clusters seems to be identified by OPTICS, which however is unable to classify a good number of moves identifying them as noise.

    A note again on the dimension of the latent space: a dimension lower than 8 seem not to give good clusters, probably too much information got lost, while a dimension higher did not improve the tactical meaning of clusters any further.
    This aspect could be studied further.

    Overall I'm satisfied with this result and I'm ready to move on to the next phase.
    """)
    return


@app.cell
def _(embedding_matrix, metadata_cols, moves_df):
    embedding_cols = [f"z_{idx:02d}" for idx in range(embedding_matrix.shape[1])]

    move_embeddings_df = pd.concat(
        [
            moves_df[metadata_cols].reset_index(drop=True),
            pd.DataFrame(embedding_matrix, columns=embedding_cols),
        ],
        axis=1,
    )

    move_embeddings_df
    return (move_embeddings_df,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Evolution
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Battle MLP

    This is the model which will take a decision on which move to play.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Inputs and Preprocessing

    | **What** |  **Size** | **Notes** |
    | -------- | -------- | --------- |
    | Moveset (own)| $4 \times 8$ dense vector | Vectors from the latent space of the AE |
    | Current PP (own) | $4$ integers | |
    | Move effectiveness score (own) | $4$ dense vector | The $\log_2$ of the multiplier due to the types of the pokemons and the moves |
    | Base Statistics (own) | $8$ dense vector | Normalized value of each stat |
    | Boosts (own) | $8$ dense vector | $\log_2$ of the multiplier each boost level gives |
    | Current HP (both) | $2$ integers | |
    | Weather and Terrain | $4 \times 2$ one-hot | |
    | Major status (both) | $7 \times 2$ one-hot | |
    | Volatile status (both) | $4 \times 2$ dense vectors | Precomputed approximation of what each status does |
    | Side conditions (both) | $3 \times 2$ dense vectors | Precomputed approximation of what each side condition does |
    | Turn number | $1$ integer | |

    This gives exactly `90` input dimensions.
    """)
    return


@app.cell(hide_code=True)
def _():
    @lru_cache()
    def _map_type_to_score(
        own_type_0: str,
        own_type_1: str | None,
        opp_type_0: str,
        opp_type_1: str | None,
        move_type: str,
    ) -> float:
        IMMUNITY_SCORE = -4.0

        type_chart: dict[str, dict[str, float]] = {
            "normal": {
                "rock": 0.5,
                "ghost": 0.0,
                "steel": 0.5,
            },
            "fire": {
                "fire": 0.5,
                "water": 0.5,
                "grass": 2.0,
                "ice": 2.0,
                "bug": 2.0,
                "rock": 0.5,
                "dragon": 0.5,
                "steel": 2.0,
            },
            "water": {
                "fire": 2.0,
                "water": 0.5,
                "grass": 0.5,
                "ground": 2.0,
                "rock": 2.0,
                "dragon": 0.5,
            },
            "electric": {
                "water": 2.0,
                "electric": 0.5,
                "grass": 0.5,
                "ground": 0.0,
                "flying": 2.0,
                "dragon": 0.5,
            },
            "grass": {
                "fire": 0.5,
                "water": 2.0,
                "grass": 0.5,
                "poison": 0.5,
                "ground": 2.0,
                "flying": 0.5,
                "bug": 0.5,
                "rock": 2.0,
                "dragon": 0.5,
                "steel": 0.5,
            },
            "ice": {
                "fire": 0.5,
                "water": 0.5,
                "grass": 2.0,
                "ice": 0.5,
                "ground": 2.0,
                "flying": 2.0,
                "dragon": 2.0,
                "steel": 0.5,
            },
            "fighting": {
                "normal": 2.0,
                "ice": 2.0,
                "poison": 0.5,
                "flying": 0.5,
                "psychic": 0.5,
                "bug": 0.5,
                "rock": 2.0,
                "ghost": 0.0,
                "dark": 2.0,
                "steel": 2.0,
                "fairy": 0.5,
            },
            "poison": {
                "grass": 2.0,
                "poison": 0.5,
                "ground": 0.5,
                "rock": 0.5,
                "ghost": 0.5,
                "steel": 0.0,
                "fairy": 2.0,
            },
            "ground": {
                "fire": 2.0,
                "electric": 2.0,
                "grass": 0.5,
                "poison": 2.0,
                "flying": 0.0,
                "bug": 0.5,
                "rock": 2.0,
                "steel": 2.0,
            },
            "flying": {
                "electric": 0.5,
                "grass": 2.0,
                "fighting": 2.0,
                "bug": 2.0,
                "rock": 0.5,
                "steel": 0.5,
            },
            "psychic": {
                "fighting": 2.0,
                "poison": 2.0,
                "psychic": 0.5,
                "dark": 0.0,
                "steel": 0.5,
            },
            "bug": {
                "fire": 0.5,
                "grass": 2.0,
                "fighting": 0.5,
                "poison": 0.5,
                "flying": 0.5,
                "psychic": 2.0,
                "ghost": 0.5,
                "dark": 2.0,
                "steel": 0.5,
                "fairy": 0.5,
            },
            "rock": {
                "fire": 2.0,
                "ice": 2.0,
                "fighting": 0.5,
                "ground": 0.5,
                "flying": 2.0,
                "bug": 2.0,
                "steel": 0.5,
            },
            "ghost": {
                "normal": 0.0,
                "psychic": 2.0,
                "ghost": 2.0,
                "dark": 0.5,
            },
            "dragon": {
                "dragon": 2.0,
                "steel": 0.5,
                "fairy": 0.0,
            },
            "dark": {
                "fighting": 0.5,
                "psychic": 2.0,
                "ghost": 2.0,
                "dark": 0.5,
                "fairy": 0.5,
            },
            "steel": {
                "fire": 0.5,
                "water": 0.5,
                "electric": 0.5,
                "ice": 2.0,
                "rock": 2.0,
                "steel": 0.5,
                "fairy": 2.0,
            },
            "fairy": {
                "fire": 0.5,
                "fighting": 2.0,
                "poison": 0.5,
                "dragon": 2.0,
                "dark": 2.0,
                "steel": 0.5,
            },
        }

        own_types = {t for t in [own_type_0, own_type_1] if t is not None}
        opp_types = {t for t in [opp_type_0, opp_type_1] if t is not None}

        stab = 1.5 if move_type in own_types else 1.0

        effectiveness = 1.0
        for defending_type in opp_types:
            effectiveness *= type_chart.get(move_type, {}).get(defending_type, 1.0)

        multiplier = stab * effectiveness

        if multiplier == 0.0:
            return IMMUNITY_SCORE

        return np.log2(multiplier)

    def map_type_to_score(
        own_type: list[str],
        opp_type: list[str],
        move_type: str,
    ):
        assert len(own_type) > 0
        assert len(opp_type) > 0

        return _map_type_to_score(
            own_type[0],
            own_type[1] if len(own_type) > 1 else None,
            opp_type[0],
            opp_type[1] if len(opp_type) > 1 else None,
            move_type,
        )

    return (map_type_to_score,)


@app.cell(hide_code=True)
def _():
    def _combine_independent_probabilities(probabilities: np.ndarray) -> float:
        if probabilities.size == 0:
            return 0.0

        probabilities = np.clip(probabilities, 0.0, 1.0)
        return float(1.0 - np.prod(1.0 - probabilities))

    ACTION_DENIAL_EFFECTS: dict[str, float] = {
        # Guaranteed loss of action.
        "flinch": 1.0,
        "mustrecharge": 1.0,
        "recharge": 1.0,
        # Probabilistic loss of action.
        "confusion": 1.0 / 3.0,
        "attract": 0.5,
        "infatuation": 0.5,
    }

    MOVE_RESTRICTION_EFFECTS: dict[str, float] = {
        # Approximate values because this function does not know the moveset.
        "taunt": 0.5,
        "encore": 0.75,
        "disable": 0.25,
        "torment": 0.25,
        "healblock": 0.25,
        "throatchop": 0.25,
        "imprison": 0.25,
        # Forced/locked move style effects.
        "choicelock": 0.75,
        "lockedmove": 0.75,
        "rollout": 0.75,
        "bide": 0.75,
        "twoturnmove": 0.75,
    }

    HP_DRIFT_EFFECTS: dict[str, float] = {
        # Harmful residual effects.
        "leechseed": -1.0 / 8.0,
        "partiallytrapped": -1.0 / 8.0,
        "partiallytrappedlock": -1.0 / 8.0,
        "wrap": -1.0 / 8.0,
        "bind": -1.0 / 8.0,
        "firespin": -1.0 / 8.0,
        "whirlpool": -1.0 / 8.0,
        "sandtomb": -1.0 / 8.0,
        "magmastorm": -1.0 / 8.0,
        "infestation": -1.0 / 8.0,
        "snaptrap": -1.0 / 8.0,
        "curse": -1.0 / 4.0,
        "nightmare": -1.0 / 4.0,
        # Beneficial residual effects.
        "aquaring": 1.0 / 16.0,
        "ingrain": 1.0 / 16.0,
    }

    PROTECTION_EFFECTS: dict[str, float] = {
        # Full protection.
        "protect": 1.0,
        "detect": 1.0,
        "kingsshield": 1.0,
        "spikyshield": 1.0,
        "banefulbunker": 1.0,
        "silktrap": 1.0,
        "burningbulwark": 1.0,
        "obstruct": 1.0,
        "maxguard": 1.0,
        # Approximate because substitute HP is unknown.
        "substitute": 0.5,
        # Semi-invulnerable two-turn move states.
        "fly": 1.0,
        "dig": 1.0,
        "dive": 1.0,
        "bounce": 1.0,
        "phantomforce": 1.0,
        "shadowforce": 1.0,
        # Charging moves are not true protection, but still encode some defensive tempo.
        "skyattack": 0.5,
        "solarbeam": 0.25,
        "solarblade": 0.25,
        # Survives lethal damage, but does not block damage.
        "endure": 0.5,
    }

    # Output order: [action_denial, move_restriction, hp_drift, protection]
    def compute_volatile_status_summary(statuses: list[str]) -> np.ndarray:
        action_denial_probs = np.array(
            [ACTION_DENIAL_EFFECTS[s] for s in statuses if s in ACTION_DENIAL_EFFECTS],
            dtype=np.float32,
        )

        move_restriction_values = np.array(
            [
                MOVE_RESTRICTION_EFFECTS[s]
                for s in statuses
                if s in MOVE_RESTRICTION_EFFECTS
            ],
            dtype=np.float32,
        )

        hp_drift_raw = sum(HP_DRIFT_EFFECTS.get(s, 0.0) for s in statuses)

        protection_values = np.array(
            [PROTECTION_EFFECTS[s] for s in statuses if s in PROTECTION_EFFECTS],
            dtype=np.float32,
        )

        action_denial = _combine_independent_probabilities(action_denial_probs)
        move_restriction = _combine_independent_probabilities(move_restriction_values)

        hp_drift = np.clip(hp_drift_raw / 0.25, -1.0, 1.0)

        protection = (
            float(np.max(protection_values)) if protection_values.size > 0 else 0.0
        )

        return np.array(
            [
                np.clip(action_denial, 0.0, 1.0),
                np.clip(move_restriction, 0.0, 1.0),
                hp_drift,
                np.clip(protection, 0.0, 1.0),
            ],
            dtype=np.float32,
        )

    return (compute_volatile_status_summary,)


@app.cell(hide_code=True)
def _():
    DEFENSIVE_SCREEN_EFFECTS: dict[str, float] = {
        "reflect": 0.35,
        "lightscreen": 0.35,
        "auroraveil": 0.5,
    }

    SPEED_SUPPORT_EFFECTS: dict[str, float] = {
        "tailwind": 1.0,
        # Usually irrelevant in no-switch 1v1 unless it already affected
        # the active Pokemon before the state snapshot.
        "stickyweb": -0.5,
    }

    STATUS_PROTECTION_EFFECTS: dict[str, float] = {
        "safeguard": 0.75,
        "mist": 0.25,
        "luckychant": 0.25,
    }

    #  Output order: [defensive_screen, speed_support, status_protection]
    def compute_side_condition_summary(conditions: list[str]) -> np.ndarray:
        defensive_screen = (
            max(DEFENSIVE_SCREEN_EFFECTS.get(c, 0.0) for c in conditions)
            if conditions
            else 0.0
        )

        speed_support = sum(SPEED_SUPPORT_EFFECTS.get(c, 0.0) for c in conditions)

        status_protection = (
            max(STATUS_PROTECTION_EFFECTS.get(c, 0.0) for c in conditions)
            if conditions
            else 0.0
        )

        return np.array(
            [
                np.clip(defensive_screen, 0.0, 1.0),
                np.clip(speed_support, -1.0, 1.0),
                np.clip(status_protection, 0.0, 1.0),
            ],
            dtype=np.float32,
        )

    return (compute_side_condition_summary,)


@app.cell(hide_code=True)
def _():
    all_weathers = {
        "raindance",
        "sunnyday",
        "sandstorm",
        "snowscape",
    }
    all_terrains = {
        "electricterrain",
        "grassyterrain",
        "mistyterrain",
        "psychicterrain",
    }
    all_major_statuses = {"brn", "par", "slp", "frz", "psn", "tox"}

    def make_onehot_map(all_states: set[str]) -> dict[str, np.ndarray]:
        encoding_map = {v: np.eye(len(all_states))[i] for i, v in enumerate(all_states)}
        encoding_map[""] = np.zeros(len(all_states))
        return encoding_map

    weathers_encoding_map = make_onehot_map(all_weathers)
    terrains_encoding_map = make_onehot_map(all_terrains)
    major_statuses_encoding_map = make_onehot_map(all_major_statuses)
    return (
        major_statuses_encoding_map,
        terrains_encoding_map,
        weathers_encoding_map,
    )


@app.function(hide_code=True)
def compute_boost_score(boost: int) -> float:
    stage = max(-6, min(6, boost))

    if boost >= 0:
        multiplier = (2 + boost) / 2
    else:
        multiplier = 2 / (2 - boost)

    return np.log2(multiplier)


@app.cell
def _(
    STAT_MAX,
    compute_side_condition_summary,
    compute_volatile_status_summary,
    major_statuses_encoding_map,
    map_type_to_score,
    metadata_cols,
    move_embeddings_df,
    moves_df,
    terrains_encoding_map,
    weathers_encoding_map,
):
    def compute_input_to_mlp(p0: sdw.PlayerState, p1: sdw.PlayerState):
        move_ids_0 = [slot["id"] for slot in p0.slots]
        return np.concatenate(
            [
                move_embeddings_df[
                    move_embeddings_df["move_identifier"].isin(move_ids_0)
                ]
                .drop(columns=metadata_cols)
                .to_numpy()
                .flatten(),
                [slot["pp"] for slot in p0.slots],
                [
                    map_type_to_score(
                        p0.pokemon["types"],
                        p1.pokemon["types"],
                        moves_df.loc[
                            moves_df["move_identifier"] == move_id, "type"
                        ].iloc[0],
                    )
                    for move_id in move_ids_0
                ],
                [v / STAT_MAX[k] for k, v in p0.pokemon["stats"].items()],
                [compute_boost_score(boost) for boost in p0.pokemon["boosts"].values()],
                [p0.pokemon["hp"], p1.pokemon["hp"]],
                weathers_encoding_map[p0.weather],
                terrains_encoding_map[p0.terrain],
                major_statuses_encoding_map[p0.pokemon["status"]],
                major_statuses_encoding_map[p1.pokemon["status"]],
                compute_volatile_status_summary(p0.pokemon["volatiles"]),
                compute_volatile_status_summary(p1.pokemon["volatiles"]),
                compute_side_condition_summary(list(p0.side_conditions.keys())),
                compute_side_condition_summary(list(p1.side_conditions.keys())),
                [p0.turn],
            ],
            dtype=np.float32,
        )

    return (compute_input_to_mlp,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Layout and Forward Pass

    The MLP will be tiny, for the initialial experimentation I picked `90 -> 16 -> 8 -> 4` where the outputs will be the logits of picking each move.
    I chose `tanh` as activation function as it bounds post-activation values to $(-1, 1)$ making the MLP less sensitive to high weights.
    """)
    return


@app.cell
def _():
    MLPParamsSlices = list[tuple[tuple[int, int], tuple[int, int]]]
    return (MLPParamsSlices,)


@app.cell
def _(MLPParamSlices, MLPParamsSlices):
    def compute_slices(layers: list[int]) -> MLPParamsSlices:
        cursor_w = 0
        cursor_b = 0
        params_slices: MLPParamSlices = []
        for i, input_size in enumerate(layers[:-1]):
            output_size = layers[i + 1]

            start_w = cursor_w
            start_b = cursor_b

            cursor_w += input_size * output_size
            cursor_b += output_size

            params_slices.append(((start_w, cursor_w), (start_b, cursor_b)))

        return params_slices

    def count_params(slices: MLPParamsSlices) -> int:
        return slices[-1][0][1] + slices[-1][1][1]

    return compute_slices, count_params


@app.cell
def _(compute_slices, count_params):
    MLP_LAYOUT = [90, 16, 8, 4]
    MLP_SLICES = compute_slices(MLP_LAYOUT)

    count_params(MLP_SLICES)
    return MLP_LAYOUT, MLP_SLICES


@app.cell
def _(MLP_SLICES):
    MLP_SLICES[0]
    return


@app.cell
def _(MLP_LAYOUT, MLP_SLICES):
    def get_params_view(params: np.ndarray, layer: int):
        slice = MLP_SLICES[layer]
        w = params[slice[0][0] : slice[0][1]].reshape(
            MLP_LAYOUT[layer], MLP_LAYOUT[layer + 1]
        )
        b = params[slice[1][0] : slice[1][1]]

        assert np.shares_memory(w, params)
        assert np.shares_memory(b, params)

        return w, b

    return (get_params_view,)


@app.cell
def _(MLP_LAYOUT, get_view):
    def mlp_forward(input: np.ndarray, params: np.ndarray) -> int:
        current = input

        for i in range(len(MLP_LAYOUT) - 2):
            w, b = get_view(params, i)
            current = np.tanh(current @ w + b)

        w, b = get_view(params, len(MLP_LAYOUT) - 2)
        current = current @ w + b

        return np.argmax(current)

    return (mlp_forward,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Defining the Agents

    We define the genome of each agent as follows:
    - `params` for the MLP, this is a flat vector which will then be sliced by looking at `MLP_SLICES`.
    - `stats` as importance scores, i.e. logits of the fraction of available stat budget.
    - `moves` is the array of the move ids.
    - `types` is a tuple of integer indicating the type(s).
    - `log_sigmas` is an array with one value per MLP layer which will control the variance of the mutation of the parameters of that layer. There will be one for weights and one for biases (so in this case $3 \times 2 = 6$).

    Note that the representation for the genotype is different than what will be fed to the MLP (the fenotype).
    """)
    return


@app.cell
def _():
    STAT_MIN = 30
    STAT_MAX = {
        "hp": 250,
        "atk": 190,
        "def": 250,
        "spa": 190,
        "spd": 250,
        "spe": 200,
    }
    STATS_TOTAL_MAX = 600

    STAT_MAX_LIST = list(STAT_MAX.values())
    STAT_KEYS = list(STAT_MAX.keys())
    return STATS_TOTAL_MAX, STAT_KEYS, STAT_MAX, STAT_MIN


@app.class_definition
class OpponentAIType(Enum):
    RANDOM = 1
    MAX_DAMAGE = 2
    SAME_MLP = 100
    MLP_FROM_PARAMS = 101


@app.cell
def _(compute_input_to_mlp, mlp_forward, moves_df, rng):
    @dataclass
    class Agent:
        params: np.ndarray
        stats: np.ndarray
        moves: np.ndarray
        types: tuple[int, int]
        log_sigmas: np.ndarray

        def get_decision_function(
            self, opponent_ai_type: OpponentAIType, userdata: Any | None = None
        ) -> sdw.MoveSelector:
            def decider(p0: sdw.PlayerState, p1: sdw.PlayerState) -> tuple[int, int]:
                input = compute_input_to_mlp(p0, p1)
                ai_move = mlp_forward(input, self.layers)

                match opponent_ai_type:
                    case OpponentAIType.RANDOM:
                        opponent_move = rng.integers(4)
                    case OpponentAIType.MAX_DAMAGE:
                        opponent_move = np.argmax(
                            [
                                moves_df.loc[
                                    moves_df["move_identifier"] == slot.id, "power"
                                ].iloc[0]
                                for slot in p1.slots
                            ]
                        )
                    case OpponentAIType.SAME_MLP:
                        opponent_move = mlp_forward(
                            compute_input_to_mlp(p1, p0), self.params
                        )
                    case OpponentAIType.MLP_FROM_PARAMS:
                        opponent_move = mlp_forward(
                            compute_input_to_mlp(p1, p0), userdata
                        )

                return ai_move, opponent_move

            return decider

        def copy(self) -> "Agent":
            return Agent(
                self.params.copy(),
                self.stats.copy(),
                self.moves.copy(),
                self.types,
                self.log_sigmas.copy(),
            )

        def __repr__(self) -> str:
            return (
                "Agent(\n"
                f"  params={np.array2string(self.params, precision=3, suppress_small=True)},\n"
                f"  stats={np.array2string(self.stats, precision=3, suppress_small=True)},\n"
                f"  moves={np.array2string(self.moves, precision=3, suppress_small=True)},\n"
                f"  types={self.types},\n"
                f"  log_sigmas={np.array2string(self.log_sigmas, precision=3, suppress_small=True)}\n"
                ")"
            )

    return (Agent,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Initialization

    - Stats can just be initialized with random numbers, since the function below will handle all cases and always output a legal distribution.
    - Types can be chosen randomly within the existing ones.
    - MLP weights are choses using Xavier Uniform initialization, since this is the recommended one when using `tanh` as activation function.
    - The first move is sampled at random, then the next ones are chosen in order to maximize the distance in the latent space, so that the moveset will be heterogeneous.
    """)
    return


@app.cell
def _(STATS_TOTAL_MAX, STAT_KEYS, STAT_MAX, STAT_MIN, rng):
    def validate_stats(stats: sdw.Stats) -> bool:
        all_below_max = all(value <= STAT_MAX[key] for key, value in stats.items())
        sum_below_max = sum(stats.values()) == STATS_TOTAL_MAX
        all_above_min = all(value >= STAT_MIN for value in stats.values())

        print(all_below_max, sum_below_max, all_above_min)

        return all_below_max and sum_below_max and all_above_min

    def genome_stats_to_integer(stats: np.ndarray) -> dict[str, int]:
        stats01 = softmax(np.asarray(stats, dtype=float))

        if len(stats01) != len(STAT_KEYS):
            raise ValueError("stats01 and keys must have the same length")

        stats01 = np.clip(stats01, 0.0, 1.0)

        mins = np.array([STAT_MIN for _ in STAT_KEYS], dtype=int)
        maxs = np.array([STAT_MAX[key] for key in STAT_KEYS], dtype=int)
        caps = maxs - mins

        min_total = int(mins.sum())
        max_total = int(maxs.sum())

        if STATS_TOTAL_MAX < min_total:
            raise ValueError("STATS_TOTAL_MAX is below the sum of minimum stats")

        if STATS_TOTAL_MAX > max_total:
            raise ValueError("STATS_TOTAL_MAX is above the sum of maximum stats")

        extra_budget = STATS_TOTAL_MAX - min_total

        if extra_budget == 0:
            return {key: int(value) for key, value in zip(STAT_KEYS, mins)}

        # Preference for extra points.
        # Multiplying by caps means that 1.0 represents "I want this stat near its own max".
        weights = stats01 * caps

        # If all normalized values are 0, there is no preference signal.
        # Fall back to distributing by available capacity.
        if weights.sum() == 0:
            weights = caps.astype(float)

        extras = np.zeros(len(STAT_KEYS), dtype=int)
        remaining = extra_budget

        active = caps > 0

        while remaining > 0:
            active_indices = np.where(active)[0]

            if len(active_indices) == 0:
                raise RuntimeError("No remaining capacity but budget is not exhausted")

            active_weights = weights[active_indices]

            if active_weights.sum() == 0:
                active_weights = caps[active_indices].astype(float)

            quotas = remaining * active_weights / active_weights.sum()

            whole = np.floor(quotas).astype(int)

            # Do not exceed remaining capacity for each stat.
            available = caps[active_indices] - extras[active_indices]
            whole = np.minimum(whole, available)

            extras[active_indices] += whole
            remaining -= int(whole.sum())

            if remaining == 0:
                break

            # Largest remainder allocation, stable by original key order.
            remainders = quotas - np.floor(quotas)

            order = sorted(
                active_indices,
                key=lambda i: (-remainders[list(active_indices).index(i)], i),
            )

            allocated_any = False

            for i in order:
                if remaining == 0:
                    break

                if extras[i] < caps[i]:
                    extras[i] += 1
                    remaining -= 1
                    allocated_any = True

            if not allocated_any:
                raise RuntimeError("Could not allocate remaining stat points")

            active = extras < caps

        values = mins + extras

        return {key: int(value) for key, value in zip(STAT_KEYS, values)}

    def random_valid_stats() -> np.ndarray:
        return rng.random(len(STAT_KEYS))

    return (random_valid_stats,)


@app.function
def xavier_uniform(fan_in, fan_out):
    # Best for tanh
    limit = np.sqrt(6 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, (fan_in, fan_out))


@app.cell
def _(metadata_cols, move_embeddings_df, rng):
    def pick_random_moveset():
        k = 4
        feature_cols = [c for c in move_embeddings_df.columns if c not in metadata_cols]

        X = move_embeddings_df[feature_cols].to_numpy()

        first = rng.integers(len(move_embeddings_df))

        selected = [first]

        min_dist = pairwise_distances(X, X[[first]]).ravel()

        for _ in range(1, k):
            next_idx = np.argmax(min_dist)
            selected.append(next_idx)

            new_dist = pairwise_distances(X, X[[next_idx]]).ravel()
            min_dist = np.minimum(min_dist, new_dist)

        return np.array(selected)
        # , move_embeddings_df.iloc[selected]["move_identifier"].tolist()

    return (pick_random_moveset,)


@app.cell
def _(
    Agent,
    MLP_LAYOUT,
    pick_random_moveset,
    random_valid_stats,
    rng,
    types_unique,
):
    def init_agent_at_random() -> Agent:
        params_parts = []
        for i, input_size in enumerate(MLP_LAYOUT[:-1]):
            output_size = MLP_LAYOUT[i + 1]

            W = xavier_uniform(input_size, output_size).flatten()
            b = np.zeros(output_size)

            params_parts.extend([W, b])

        params = np.concatenate(params_parts)

        stats = random_valid_stats()

        types = (
            rng.integers(len(types_unique), dtype=int),
            rng.integers(len(types_unique), dtype=int),
        )

        moves = pick_random_moveset()

        log_sigmas = np.log(
            np.ones((len(MLP_LAYOUT) - 1) * 2) * 0.05
        )  # initially set sigma = 0.05 for all layers

        return Agent(params, stats, moves, types, log_sigmas)

    return (init_agent_at_random,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Mutations
    """)
    return


@app.cell
def _(Agent):
    def mutate_sigma(agent: Agent, tau=0.10):
        LOG_SIGMA_MIN = np.log(1e-4)
        LOG_SIGMA_MAX = np.log(0.5)

        agent.log_sigmas = agent.log_sigmas.copy() + np.clip(
            np.random.normal(0, tau, size=agent.log_sigmas.shape),
            LOG_SIGMA_MIN,
            LOG_SIGMA_MAX,
        )

        return agent

    return (mutate_sigma,)


@app.cell
def _(Agent, MLP_LAYOUT, get_params_view):
    def mutate_params(agent: Agent, indpb=0.05):
        params = agent.params.copy()
        sigmas = np.exp(agent.log_sigmas)

        for layer in range(len(MLP_LAYOUT) - 1):
            w, b = get_params_view(params, layer)
            sigma_w = sigmas[layer * 2]
            sigma_b = sigmas[layer * 2 + 1]

            mask_w = np.random.random(w.shape) < indpb
            mask_b = np.random.random(b.shape) < indpb

            w[mask_w] += np.random.normal(0, sigma_w, size=mask_w.sum())
            b[mask_b] += np.random.normal(0, sigma_b, size=mask_b.sum())

            assert np.shares_memory(w, params)
            assert np.shares_memory(b, params)

        agent.params = np.clip(params, -5, 5)
        return agent

    return (mutate_params,)


@app.cell
def _(Agent, rng, types_unique):
    def mutate_type(agent: Agent, p=0.25, p_collapse=0.05):
        t1, t2 = agent.types

        if np.random.random() < p:
            t1 = rng.integers(len(types_unique))

        if np.random.random() < p:
            t2 = rng.integers(len(types_unique))

        if np.random.random() < p_collapse:
            if np.random.random() < 0.5:
                t2 = t1
            else:
                t1 = t2

        agent.types = (t1, t2)
        return agent

    return (mutate_type,)


@app.cell
def _(metadata_cols, move_embeddings_df):
    SIGMA_MOVE_CHANGE = 0.5

    _feature_cols = [c for c in move_embeddings_df.columns if c not in metadata_cols]

    _X = move_embeddings_df[_feature_cols].to_numpy()

    move_dist_matrix = cdist(_X, _X, metric="euclidean")
    _logits = -(move_dist_matrix**2) / (2.0 * SIGMA_MOVE_CHANGE**2)
    np.fill_diagonal(_logits, -np.inf)

    move_local_change_p = softmax(_logits, axis=1)

    #  pd.Series(move_local_change_p[5][move_local_change_p[5] > 0.01])
    return (move_local_change_p,)


@app.cell
def _(Agent, move_embeddings_df, move_local_change_p, rng):
    def mutate_moveset(agent: Agent, p_replace=0.25, p_jump=0.2):
        moves = agent.moves.copy()

        for i in range(len(moves)):
            if np.random.random() < p_replace:
                current = moves[i]

                if np.random.random() > p_jump:
                    probs = move_local_change_p[current]
                    moves[i] = rng.choice(len(move_embeddings_df), p=probs)
                else:
                    moves[i] = rng.integers(len(move_embeddings_df))

        agent.moves = moves
        return agent

    return (mutate_moveset,)


@app.cell
def _(Agent):
    def mutate_stats(agent: Agent, sigma=0.1, indpb=0.5):
        stats = agent.stats.copy()

        mask = np.random.random(stats.shape) < indpb
        stats[mask] += np.random.normal(0, sigma, size=mask.sum())

        agent.stats = stats
        return agent

    return (mutate_stats,)


@app.cell
def _(
    Agent,
    mutate_moveset,
    mutate_params,
    mutate_sigma,
    mutate_stats,
    mutate_type,
):
    def mutate_all(agent: Agent):
        if np.random.random() < 0.40:
            mutate_moveset(agent)

        if np.random.random() < 0.10:
            mutate_type(agent)

        if np.random.random() < 0.70:
            mutate_stats(agent)

        if np.random.random() < 0.8:
            mutate_sigma(agent)

        if np.random.random() < 0.95:
            mutate_params(agent)

        return agent

    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Crossover
    """)
    return


@app.cell
def _(Agent, rng):
    def mate(a: Agent, b: Agent):
        # Moveset crossover
        if np.random.random() < 0.5:
            a.moves = a.moves.copy()
            b.moves = b.moves.copy()

            i = rng.integers(len(a.moves))
            j = rng.integers(len(b.moves))

            a.moves[i], b.moves[j] = b.moves[j], a.moves[i]

        # Type crossover
        if np.random.random() < 0.5:
            i = 0 if np.random.random() < 0.5 else 1
            j = 0 if np.random.random() < 0.5 else 1

            t_a = (a.types[-(i - 1)], b.types[j])
            t_b = (b.types[-(j - 1)], a.types[i])

            a.types = t_a
            b.types = t_b

        # Stat crossover
        if np.random.random() < 0.5:
            alpha = np.random.random()
            xa = a.stats.copy()
            xb = b.stats.copy()
            a.stats = alpha * xa + (1 - alpha) * xb
            b.stats = alpha * xb + (1 - alpha) * xa

        # MLP crossover
        if np.random.random() < 0.2:
            a.params, b.params = b.params.copy(), a.params.copy()

        return a, b

    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Algorithm
    """)
    return


@app.cell
def _(Agent, init_agent_at_random):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", Agent, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    toolbox.register("individual", init_agent_at_random)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    return


@app.cell
def _():
    return


@app.cell
def _(seed):
    def first_move_ai(p0: sdw.PlayerState, p1: sdw.PlayerState) -> tuple[int, int]:
        print(p0)

        return (0, 0)

    ai = {
        "species": "Pikachu",
        "types": ["Electric"],
        "stats": {
            "hp": 250,
            "atk": 150,
            "def": 100,
            "spa": 120,
            "spd": 100,
            "spe": 180,
        },
        "moves": ["thunderbolt", "irontail", "quickattack", "thunderwave"],
    }

    configs = [
        sdw.BattleConfig(
            ai=ai,
            opponent={"type": "hardcoded", "species": "Garchomp"},
            move_selector=first_move_ai,
            seed=seed,
        ),
        sdw.BattleConfig(
            ai=ai,
            opponent={"type": "hardcoded"},
            move_selector=first_move_ai,
            seed=seed,
        ),
        sdw.BattleConfig(
            ai=ai,
            opponent={"type": "random"},
            move_selector=first_move_ai,
            seed=seed,
        ),
        sdw.BattleConfig(
            ai=ai,
            opponent={"type": "random"},
            move_selector=first_move_ai,
            seed=seed,
        ),
    ]

    with sdw.ShowdownPool(max_size=1) as pool:
        results = pool.run_battles(configs)

    for i, r in enumerate(results):
        print(
            f"Battle {i}: {r.winner} won in {r.turns} turns "
            f"(p0 HP: {r.player_hp}, p1 HP: {r.opponent_hp})"
        )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Fitness

    The fitness will be computed using this formula
    $$
    \begin{align*}
    \text{fitness for 1 battle} = &\, 3000 \cdot \mathbb I(\text{battle won}) \\
    & + 100 \cdot \text{own remaining HP} \\
    & - 100 \cdot \text{opponent remaining HP} \\
    & - 50  \cdot \text{number of turns}
    \end{align*}
    $$
    where $\mathbb I$ is the indicator function.

    The model will battle against various opponents with various degrees of difficulty.
    The total fitness will be the sum of the fitness scores of each battle.
    """)
    return


if __name__ == "__main__":
    app.run()

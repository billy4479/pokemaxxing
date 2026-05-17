import marimo

__generated_with = "0.21.1"
app = marimo.App(width="full")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import precision_recall_fscore_support, silhouette_score
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
    from torch import nn
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, TensorDataset
    from tqdm import tqdm
    from umap import UMAP

    from lib.dataset.queries import build_moves_table
    from pathlib import Path


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

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device, seed


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # PokéMaxxing - What is the ideal Pokémon?

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
    - An ability within the existing ones;
    - 2 types within the existing ones (which can be the same).

    Most importantly, to evaluate the fitness of each agent they will need to fight against other pokémons, so I will evolve a small MLP which will choose what to do in the fight (I will refer to it as the "battle MLP").
    While the topology of the battle MLP will be fixed, the weights will evolve alongside the other genes (so no gradient descent here).
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Luckily, since pokémon is a very famous franchise, a lot of great tools already exist.

    - I used the [PokéAPI](https://github.com/PokeAPI/pokeapi/) dataset as the main dataset for this project. This dataset is licensed under the [BSD-3-Clause License](https://github.com/PokeAPI/pokeapi/blob/master/LICENSE.md).
    - Another dataset I used was the [Pokémon Database](https://pokemondb.net/sun-moon/zmoves), expecially for Z-Moves.
    - The battle logic is implemented by the awesome [Pokémon Showdown](https://github.com/smogon/pokemon-showdown) library licensed under the [MIT License](https://github.com/smogon/pokemon-showdown). I had to wrap this library in a special NodeJS+Python harness in order to facilitate the use with this project.

    While most of the code for this project is written by me, I was assisted by AI, expecially in the prototyping phase.
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
        if col
        not in metadata_cols + categorical_cols + flag_cols + stat_change_cols
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
    numeric_scaled_df = ((numeric_df - numeric_mean) / numeric_std).astype(
        np.float32
    )
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

    column_to_index = {
        column: index for index, column in enumerate(feature_df.columns)
    }

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
            group_name: group["size"]
            for group_name, group in feature_groups.items()
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
    """)
    return


@app.cell
def _(categorical_feature_groups, feature_groups, group_loss_weights):
    class MoveAutoencoder(nn.Module):
        def __init__(self, input_dim, latent_dim=16, hidden_dims=(256, 128)):
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
        use_cache = False,
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
                        key: value / dataset_size
                        for key, value in epoch_totals.items()
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
        reconstruction_logits_matrix = (
            reconstruction_logits_eval.detach().cpu().numpy()
        )

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
        feature_df, latent_dim=8, epochs=512 * 2, loss_weights=group_loss_weights, use_cache=True,
    )

    if training_history is not None:
        training_history.tail(10)
    else:
        print("Weights loaded from cache.")
    return embedding_matrix, reconstruction_logits_matrix, training_history


@app.cell
def _(device, training_history):
    if training_history is not None:
        final_training_row = training_history.iloc[-1]
    
        training_summary = {
            "device": str(device),
            "epochs": int(training_history["epoch"].max()),
            "final_loss": float(final_training_row["loss"]),
            "best_loss": float(training_history["loss"].min()),
            "final_recon_numeric": float(final_training_row["recon_numeric"]),
            "final_recon_stat_change": float(final_training_row["recon_stat_change"]),
            "final_recon_categorical": float(final_training_row["recon_categorical"]),
            "final_recon_flags": float(final_training_row["recon_flags"]),
        }

        training_summary

    return


@app.cell
def _(training_history):
    if training_history is not None:
        _fig, _ax = plt.subplots()
    
        _ax.plot(
            training_history["epoch"],
            training_history["loss"],
            label="total",
            linewidth=2.5,
        )
    
        _ax.plot(
            training_history["epoch"],
            training_history["recon_numeric"],
            label="numeric",
            linewidth=1.5,
        )
    
        _ax.plot(
            training_history["epoch"],
            training_history["recon_stat_change"],
            label="stat_change",
            linewidth=1.5,
        )
    
        _ax.plot(
            training_history["epoch"],
            training_history["recon_categorical"],
            label="categorical",
            linewidth=1.5,
        )
    
        _ax.plot(
            training_history["epoch"],
            training_history["recon_flags"],
            label="flags",
            linewidth=1.5,
        )
    
        _fig.suptitle("Training Progress")
        _ax.set_xlabel("Epoch")
        _ax.set_ylabel("Loss")
        _ax.grid(alpha=0.25)
        _ax.legend()
    
        _fig.tight_layout()
        _fig
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
        y_pred = (
            reconstruction_probability_df.iloc[:, group_slice].to_numpy() >= 0.5
        )

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
    _plot_df["label"] = _plot_df["group"] + "\n" + _plot_df["metric"]

    _fig, _ax = plt.subplots(figsize=(12, 5))

    _ax.bar(
        _plot_df["label"],
        _plot_df["value"],
    )

    _ax.set_title("Reconstruction Evaluation Summary")
    _ax.set_ylabel("Metric value")
    _ax.tick_params(axis="x", rotation=45)
    _ax.grid(axis="y", alpha=0.25)

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
    A comment on the choice of hyperparameters:
    - The learning rate was made low and the number of epochs high as we don't care if the autoencoder overfits or if the training takes longer, we just want the best result with the least possible number of dimensions in the latent space.
    - Flags used to have a worse reconstruction than other parameters, so I bumped their loss weight so that the model would optimize for them more aggressively.
    - I experimented a lot with the dimension of the latent space. It seems like a higher number of dimensions may help when the number of epochs is low, however with enough training the reconstruction is fine even with as little as 8 dimensions. Lower than 8 we start losing some structure, but I will comment on this again later in the clustering section.
    """)
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
def plot_umap_pca_2d(
    umap, pca, labels, get_mask, title, legend_title="", legend=True
):
    fig, (ax1, ax2) = plt.subplots(1, 2)

    fig.suptitle(title)
    fig.tight_layout()

    cmap = plt.get_cmap("tab20")
    colors_per_label = {
        label: cmap(i % cmap.N) for i, label in enumerate(labels)
    }

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
    return


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
    kmeans_labels, k = best_kmeans_silhouette(embedding_matrix, max_k=48)
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
    import plotly.express as px

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
    return


if __name__ == "__main__":
    app.run()

import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch
    from sklearn.cluster import DBSCAN, OPTICS, KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
    from torch import nn
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, TensorDataset
    from tqdm.auto import tqdm
    from umap import UMAP

    from lib.dataset.queries import build_moves_table


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Encoding moves

    Pokémon moves are increadibly rich in information content: they have many numerical statistics and, in particular, a lot of categorical features such as the type or the effect.

    In particular the effect is problematic as they are so complicated that no dataset even attempts to describe what each effect does numerically, preferring a text description instead.

    In this notebook we explore how to encode moves in a lower dimensional space trying to preserve meaning: moves which gets used in similar situations should be close together in the resulting embedding, so that this embedding can be used as a preprocessing step in other ML projects.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    I used the [PokéAPI](https://github.com/PokeAPI/pokeapi/) dataset as the main source of truth for this project. This dataset is licensed under the [BSD-3-Clause License](https://github.com/PokeAPI/pokeapi/blob/master/LICENSE.md).

    Another dataset I used was the [Pokémon Database](https://pokemondb.net/sun-moon/zmoves), expecially for Z-Moves.

    I started by importing the moves in a format I can use. Some less relevant columns were already removed at this stage, moreover Z-Moves are filtered out as they add a non-trivial amount of complexity.
    """)
    return


@app.cell
def _():
    moves_df = build_moves_table().reset_index(drop=True)
    moves_df
    return (moves_df,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We start by reprenting this table as a matrix, where some entries are one-hot encoded.
    """)
    return


@app.cell
def _(moves_df):
    metadata_cols = ["move_id", "move_identifier"]
    categorical_cols = ["type", "damage_class", "target", "effect"]
    excluded_numeric_cols = {"generation_id", "target_id", "name"}

    numeric_cols = [
        col
        for col in moves_df.columns
        if col not in metadata_cols + categorical_cols
        and col not in excluded_numeric_cols
        and pd.api.types.is_numeric_dtype(moves_df[col])
    ]

    assert "generation_id" not in numeric_cols

    numeric_df = moves_df[numeric_cols].astype(np.float32)
    numeric_mean = numeric_df.mean(axis=0)
    numeric_std = numeric_df.std(axis=0, ddof=0).replace(0.0, 1.0)
    numeric_scaled_df = ((numeric_df - numeric_mean) / numeric_std).astype(
        np.float32
    )

    categorical_df = pd.get_dummies(
        moves_df[categorical_cols].astype("category"),
        prefix=categorical_cols,
        dtype=np.float32,
    )

    feature_df = pd.concat([numeric_scaled_df, categorical_df], axis=1)

    one_hot_blocks = {
        categorical_col: [
            encoded_col
            for encoded_col in categorical_df.columns
            if encoded_col.startswith(f"{categorical_col}_")
        ]
        for categorical_col in categorical_cols
    }

    layout_rows = [
        {
            "block": "numeric",
            "start_idx": 0,
            "end_idx": len(numeric_cols) - 1,
            "width": len(numeric_cols),
            "representation": "z-scored float32",
        }
    ]
    cursor = len(numeric_cols)
    categorical_slices = {}
    for _categorical_col in categorical_cols:
        _width = len(one_hot_blocks[_categorical_col])
        categorical_slices[_categorical_col] = (cursor, cursor + _width)
        layout_rows.append(
            {
                "block": _categorical_col,
                "start_idx": cursor,
                "end_idx": cursor + _width - 1,
                "width": _width,
                "representation": "one-hot binary float32",
            }
        )
        cursor += _width

    input_layout_df = pd.DataFrame(layout_rows)
    input_spec = {
        "input_dim": int(feature_df.shape[1]),
        "numeric_dim": len(numeric_cols),
        "categorical_dim": int(feature_df.shape[1] - len(numeric_cols)),
        "excluded": ["generation_id", "target_id"],
    }

    input_spec
    input_layout_df
    return (
        categorical_cols,
        categorical_slices,
        feature_df,
        metadata_cols,
        numeric_cols,
        one_hot_blocks,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Autoencoder

    The first approach I tried is to train an AE and study its latent space.
    """)
    return


@app.cell
def _(feature_df):
    batch_size = 64
    x = torch.tensor(feature_df.to_numpy(), dtype=torch.float32)
    dataset = TensorDataset(x)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    input_dim = x.shape[1]
    return dataloader, dataset, input_dim, x


@app.cell
def _():
    class MoveAutoencoder(nn.Module):
        def __init__(self, input_dim, latent_dim=16, hidden_dims=(256, 128)):
            super().__init__()

            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dims[0]),
                nn.ReLU(),
                nn.Linear(hidden_dims[0], hidden_dims[1]),
                nn.ReLU(),
                nn.Linear(hidden_dims[1], latent_dim),
                nn.LayerNorm(latent_dim),
            )

            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dims[1]),
                nn.ReLU(),
                nn.Linear(hidden_dims[1], hidden_dims[0]),
                nn.ReLU(),
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


    def ae_loss(
        _reconstruction_logits,
        _inputs,
        categorical_cols,
        categorical_loss_weights,
        categorical_slices,
        n_numeric,
    ):
        reconstructed_numeric = _reconstruction_logits[:, :n_numeric]

        target_numeric = _inputs[:, :n_numeric]

        recon_numeric = F.mse_loss(
            reconstructed_numeric,
            target_numeric,
            reduction="mean",
        )

        recon_categorical = torch.zeros((), device=_inputs.device)
        for _categorical_col in categorical_cols:
            _start, _end = categorical_slices[_categorical_col]
            _block_logits = _reconstruction_logits[:, _start:_end]
            _block_target_indices = torch.argmax(_inputs[:, _start:_end], dim=1)
            _weight = float(categorical_loss_weights.get(_categorical_col, 1.0))
            recon_categorical = recon_categorical + _weight * F.cross_entropy(
                _block_logits,
                _block_target_indices,
                reduction="mean",
            )

        total = recon_numeric + recon_categorical

        return total, recon_numeric, recon_categorical

    return MoveAutoencoder, ae_loss


@app.cell
def _():
    seed = 42
    latent_dim = 8*2
    epochs = 128
    learning_rate = 8e-4

    categorical_loss_weights = {
        "type": 2.0,
        "damage_class": 1.0,
        "target": 1.0,
        "effect": 0.5,
    }
    return categorical_loss_weights, epochs, latent_dim, learning_rate, seed


@app.cell
def _(
    MoveAutoencoder,
    ae_loss,
    categorical_cols,
    categorical_loss_weights,
    categorical_slices,
    dataloader,
    dataset,
    epochs,
    input_dim,
    latent_dim,
    learning_rate,
    numeric_cols,
    seed,
):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MoveAutoencoder(input_dim=input_dim, latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    n_numeric = len(numeric_cols)
    history_rows = []

    _epoch_progress = tqdm(range(1, epochs + 1), desc="Training AE", unit="epoch")
    for epoch in _epoch_progress:
        model.train()

        epoch_total = 0.0
        epoch_numeric = 0.0
        epoch_categorical = 0.0

        for (batch,) in dataloader:
            batch = batch.to(device)

            optimizer.zero_grad()
            _reconstruction_logits_batch, _latent_batch = model(batch)
            loss, recon_numeric, recon_categorical = ae_loss(
                _reconstruction_logits_batch,
                batch,
                categorical_cols=categorical_cols,
                categorical_loss_weights=categorical_loss_weights,
                categorical_slices=categorical_slices,
                n_numeric=n_numeric,
            )
            loss.backward()
            optimizer.step()

            batch_count = batch.shape[0]
            epoch_total += loss.item() * batch_count
            epoch_numeric += recon_numeric.item() * batch_count
            epoch_categorical += recon_categorical.item() * batch_count

        dataset_size = len(dataset)
        history_rows.append(
            {
                "epoch": epoch,
                "loss": epoch_total / dataset_size,
                "recon_numeric": epoch_numeric / dataset_size,
                "recon_categorical": epoch_categorical / dataset_size,
            }
        )
        _epoch_progress.set_postfix(
            loss=f"{history_rows[-1]['loss']:.4f}",
            recon_num=f"{history_rows[-1]['recon_numeric']:.4f}",
            recon_cat=f"{history_rows[-1]['recon_categorical']:.4f}",
        )

    history_df = pd.DataFrame(history_rows)
    history_df.tail(10)
    return device, history_df, model, n_numeric


@app.cell
def _(device, history_df):
    training_summary = {
        "device": str(device),
        "epochs": int(history_df["epoch"].max()),
        "final_loss": float(history_df.iloc[-1]["loss"]),
        "best_loss": float(history_df["loss"].min()),
    }
    training_summary
    return


@app.cell
def _(device, model, x):
    model.eval()
    with torch.no_grad():
        _latent_embedding = model.encode(x.to(device))
        _reconstruction_logits_eval = model.decode(_latent_embedding)
    latent_embedding = _latent_embedding
    eval_reconstruction_logits = _reconstruction_logits_eval
    return eval_reconstruction_logits, latent_embedding


@app.cell
def _(
    categorical_cols,
    eval_reconstruction_logits,
    n_numeric,
    numeric_cols,
    one_hot_blocks,
    x,
):
    reconstruction_logits_cpu = eval_reconstruction_logits.detach().cpu()
    x_cpu = x.detach().cpu()

    numeric_target = x_cpu[:, :n_numeric]
    numeric_reconstruction = reconstruction_logits_cpu[:, :n_numeric]
    numeric_mse = torch.mean((numeric_reconstruction - numeric_target) ** 2, dim=0)

    numeric_reconstruction_df = pd.DataFrame(
        {
            "feature": numeric_cols,
            "mse": numeric_mse.numpy(),
        }
    ).sort_values("mse", ascending=False)

    block_accuracies = []
    offset = n_numeric
    for _categorical_col in categorical_cols:
        _width = len(one_hot_blocks[_categorical_col])
        target_block = x_cpu[:, offset : offset + _width]
        prediction_block = reconstruction_logits_cpu[:, offset : offset + _width]

        target_idx = torch.argmax(target_block, dim=1)
        prediction_idx = torch.argmax(prediction_block, dim=1)
        accuracy = torch.mean((prediction_idx == target_idx).float()).item()

        block_accuracies.append(
            {
                "block": _categorical_col,
                "accuracy": accuracy,
                "width": _width,
            }
        )
        offset += _width

    categorical_reconstruction_df = pd.DataFrame(block_accuracies)
    (categorical_reconstruction_df, numeric_reconstruction_df)
    return categorical_reconstruction_df, numeric_reconstruction_df


@app.cell
def _(history_df):
    _fig, _ax = plt.subplots(figsize=(9, 4))

    _ax.plot(history_df["epoch"], history_df["loss"], label="total", linewidth=2)
    _ax.plot(
        history_df["epoch"],
        history_df["recon_numeric"],
        label="recon_numeric",
        linewidth=1.5,
    )
    _ax.plot(
        history_df["epoch"],
        history_df["recon_categorical"],
        label="recon_categorical",
        linewidth=1.5,
    )
    _ax.set_title("Training Progress")
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
    The reconstruction accuracy of the AE is very good: almost no loss at all even on a very little latent space.
    """)
    return


@app.cell
def _(categorical_reconstruction_df, numeric_reconstruction_df):
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4))

    _top_numeric = numeric_reconstruction_df.nlargest(10, "mse").iloc[::-1]
    _axes[0].barh(_top_numeric["feature"], _top_numeric["mse"], color="tab:red")
    _axes[0].set_title("Top Numeric Reconstruction Errors")
    _axes[0].set_xlabel("MSE")
    _axes[0].grid(axis="x", alpha=0.25)

    _axes[1].bar(
        categorical_reconstruction_df["block"],
        categorical_reconstruction_df["accuracy"],
        color="tab:blue",
    )
    _axes[1].set_ylim(0.0, 1.0)
    _axes[1].set_title("Categorical Block Reconstruction Accuracy")
    _axes[1].set_ylabel("Accuracy")
    _axes[1].axhline(
        0.7,
        color="tab:green",
        linestyle="--",
        linewidth=1.5,
        label="type target = 0.7",
    )
    _axes[1].grid(axis="y", alpha=0.25)
    _axes[1].legend()

    _fig.tight_layout()
    _fig
    return


@app.cell
def _(latent_embedding):
    embedding_matrix = latent_embedding.detach().cpu().numpy()
    return (embedding_matrix,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Understanding the structure of the latent space

    Even if the reconstruction accuracy is good it doesn't mean that the latent space is in a good shape.
    We try to plot the moves on a 2D UMAP and PCA.

    Sadly it seems like there are no clusters.
    """)
    return


@app.cell
def _(embedding_matrix, seed):
    _umap_model = UMAP(
        n_components=2,
        n_neighbors=20,
        min_dist=0.05,
        metric="euclidean",
        random_state=seed,
    )
    umap_embedding_2d = _umap_model.fit_transform(embedding_matrix)
    return (umap_embedding_2d,)


@app.cell
def _(embedding_matrix):
    _pca_model = PCA(n_components=2)
    pca_embedding_2d = _pca_model.fit_transform(embedding_matrix)
    return (pca_embedding_2d,)


@app.cell
def _(latent_embedding):
    min_samples = 10
    eps_percentile = 90
    k_range = range(2, 13)
    random_state = 42

    _embedding_matrix = latent_embedding.detach().cpu().numpy()
    _scaled = StandardScaler().fit_transform(_embedding_matrix)

    _neighbors = NearestNeighbors(n_neighbors=min_samples)
    _neighbors.fit(_scaled)
    _distances, _ = _neighbors.kneighbors(_scaled)
    _k_distances = np.sort(_distances[:, -1])
    _eps = float(np.percentile(_k_distances, eps_percentile))

    _dbscan = DBSCAN(eps=_eps, min_samples=min_samples)
    _dbscan_labels = _dbscan.fit_predict(_scaled)

    _optics = OPTICS(min_samples=min_samples, xi=0.05, min_cluster_size=0.05)
    _optics_labels = _optics.fit_predict(_scaled)

    _silhouette_scores = []
    _best_k = None
    _best_score = -1.0
    for _k in k_range:
        _kmeans = KMeans(n_clusters=_k, random_state=random_state, n_init="auto")
        _labels = _kmeans.fit_predict(_scaled)
        _score = silhouette_score(_scaled, _labels)
        _silhouette_scores.append({"k": _k, "silhouette": _score})
        if _score > _best_score:
            _best_score = _score
            _best_k = _k

    _best_k = _best_k or 2
    _kmeans = KMeans(n_clusters=_best_k, random_state=random_state, n_init="auto")
    _kmeans_labels = _kmeans.fit_predict(_scaled)

    cluster_results = {
        "dbscan": {
            "labels": _dbscan_labels,
            "params": {
                "min_samples": min_samples,
                "eps_percentile": eps_percentile,
                "eps": _eps,
            },
        },
        "optics": {
            "labels": _optics_labels,
            "params": {
                "min_samples": min_samples,
                "xi": 0.05,
                "min_cluster_size": 0.05,
            },
        },
        "kmeans": {
            "labels": _kmeans_labels,
            "params": {
                "k": _best_k,
                "silhouette": _best_score,
                "k_range": f"{min(k_range)}-{max(k_range)}",
            },
        },
    }

    _overview_rows = []
    _size_rows = []
    for _method, _payload in cluster_results.items():
        _labels = _payload["labels"]
        _unique = set(_labels)
        _clusters = len(_unique - {-1})
        _noise = int((_labels == -1).sum())
        _overview_rows.append(
            {
                "method": _method,
                "clusters": _clusters,
                "noise_points": _noise,
                "params": _payload["params"],
            }
        )

        _counts = pd.Series(_labels, name="cluster").value_counts(dropna=False)
        for _label, _count in _counts.items():
            _size_rows.append(
                {"method": _method, "cluster": int(_label), "count": int(_count)}
            )

    clustering_overview_df = pd.DataFrame(_overview_rows)
    cluster_sizes_df = pd.DataFrame(_size_rows)
    silhouette_scores_df = pd.DataFrame(_silhouette_scores)

    (clustering_overview_df, cluster_sizes_df, silhouette_scores_df)
    return (cluster_results,)


@app.cell
def _(moves_df, pca_embedding_2d, umap_embedding_2d):
    _types_sorted = sorted(moves_df["type"].unique())
    _cmap = plt.get_cmap("tab20")
    _type_colors = {
        _move_type: _cmap(index % _cmap.N)
        for index, _move_type in enumerate(_types_sorted)
    }

    _fig, _axes = plt.subplots(1, 2, figsize=(12, 5))
    for _move_type in _types_sorted:
        _mask = (moves_df["type"] == _move_type).to_numpy()
        _axes[0].scatter(
            umap_embedding_2d[_mask, 0],
            umap_embedding_2d[_mask, 1],
            s=20,
            alpha=0.7,
            color=_type_colors[_move_type],
            label=_move_type,
        )
        _axes[1].scatter(
            pca_embedding_2d[_mask, 0],
            pca_embedding_2d[_mask, 1],
            s=20,
            alpha=0.7,
            color=_type_colors[_move_type],
            label=_move_type,
        )

    _axes[0].set_title("Embedding UMAP (colored by type)")
    _axes[0].set_xlabel("UMAP 1")
    _axes[0].set_ylabel("UMAP 2")
    _axes[0].grid(alpha=0.25)

    _axes[1].set_title("Embedding PCA (colored by type)")
    _axes[1].set_xlabel("PC1")
    _axes[1].set_ylabel("PC2")
    _axes[1].grid(alpha=0.25)
    _axes[1].legend(title="type", ncol=2, fontsize=8)

    _fig.tight_layout()
    _fig
    return


@app.cell
def _(cluster_results, pca_embedding_2d, umap_embedding_2d):
    _methods = list(cluster_results.keys())
    _row_count = len(_methods)
    _fig, _axes = plt.subplots(_row_count, 2, figsize=(12, 5 * _row_count))
    if _row_count == 1:
        _axes = np.array([_axes])

    for _row_index, _method in enumerate(_methods):
        _labels = cluster_results[_method]["labels"]
        _unique_labels = sorted(set(_labels))
        _cmap = plt.get_cmap("tab20")

        _label_colors = {}
        _color_index = 0
        for _label in _unique_labels:
            if _label == -1:
                _label_colors[_label] = (0.6, 0.6, 0.6, 0.6)
            else:
                _label_colors[_label] = _cmap(_color_index % _cmap.N)
                _color_index += 1

        for _label in _unique_labels:
            _mask = _labels == _label
            _axes[_row_index, 0].scatter(
                umap_embedding_2d[_mask, 0],
                umap_embedding_2d[_mask, 1],
                s=20,
                alpha=0.7,
                color=_label_colors[_label],
                label=f"cluster {_label}",
            )
            _axes[_row_index, 1].scatter(
                pca_embedding_2d[_mask, 0],
                pca_embedding_2d[_mask, 1],
                s=20,
                alpha=0.7,
                color=_label_colors[_label],
                label=f"cluster {_label}",
            )

        _axes[_row_index, 0].set_title(f"Embedding UMAP (colored by {_method})")
        _axes[_row_index, 0].set_xlabel("UMAP 1")
        _axes[_row_index, 0].set_ylabel("UMAP 2")
        _axes[_row_index, 0].grid(alpha=0.25)

        _axes[_row_index, 1].set_title(f"Embedding PCA (colored by {_method})")
        _axes[_row_index, 1].set_xlabel("PC1")
        _axes[_row_index, 1].set_ylabel("PC2")
        _axes[_row_index, 1].grid(alpha=0.25)
        _axes[_row_index, 1].legend(ncol=2, fontsize=8)

    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Even if there are no clusters we try a more manual approach: we sample some moves at random from the dataset and look at their neighbors.
    I then asked some "pokemon-informed" friends if the close by moves are actually related in some way.
    """)
    return


@app.cell
def _(embedding_matrix, moves_df):
    def get_neightbours(move, count=10, metric="cosine", output_df=False):
        row_count = len(embedding_matrix)
        query_str = move.strip().lower()
        identifier_match = moves_df["move_identifier"].str.lower() == query_str
        name_match = (
            moves_df["name"].str.lower() == query_str
            if "name" in moves_df.columns
            else pd.Series(False, index=moves_df.index)
        )
        indices = moves_df.index[identifier_match | name_match].to_numpy()

        if indices.size == 0:
            raise Exception(f"move query '{move}' not found")

        query_index = int(indices[0])
        k = min(count + 1, row_count)
        nn = NearestNeighbors(n_neighbors=k, metric=metric)
        nn.fit(embedding_matrix)
        distances, neighbors = nn.kneighbors(
            embedding_matrix[query_index : query_index + 1]
        )

        neighbor_indices = neighbors[0].tolist()
        neighbor_distances = distances[0].tolist()

        results = []
        for neighbor_idx, distance in zip(neighbor_indices, neighbor_distances):
            if neighbor_idx == query_index:
                continue
            if output_df:
                results.append(
                    {
                        "rank": len(results) + 1,
                        "distance": float(distance),
                        **(
                            moves_df.loc[neighbor_idx]
                            .drop(
                                [
                                    "move_id",
                                    "move_identifier",
                                    "generation_id",
                                    "target_id",
                                ]
                            )
                            .to_dict()
                        ),
                    }
                )
            else:
                results.append(
                    {
                        "move": moves_df.loc[neighbor_idx, "move_identifier"],
                        "distance": float(distance),
                    }
                )

            if len(results) >= count:
                break

        if output_df:
            df = pd.DataFrame(results)[
                [
                    "rank",
                    "distance",
                    "name",
                    "type",
                    "power",
                    "pp",
                    "target",
                    "accuracy",
                    "priority",
                    "effect_chance",
                    "effect",
                    "turns",
                    "damage_class",
                    "drain",
                    "healing",
                    "crit_rate",
                    "ailment_chance",
                    "flinch_chance",
                    "stat_chance",
                    "stat_change_accuracy",
                    "stat_change_attack",
                    "stat_change_defense",
                    "stat_change_evasion",
                    "stat_change_special_attack",
                    "stat_change_special_defense",
                    "stat_change_speed",
                ]
            ]
            df.style.set_caption(move)
            return df
        return results

    return (get_neightbours,)


@app.cell
def _(get_neightbours, moves_df):
    {
        move: get_neightbours(move, output_df=False, metric="euclidean")
        for move in moves_df["move_identifier"].sample(n=10)
    }
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


@app.cell
def _(get_neightbours):
    lol = [
        "recover",
        "protect",
        "toxic",
        "swords-dance",
        "nasty-plot",
        "thunder-wave",
        "will-o-wisp",
        "u-turn",
        "volt-switch",
        "bullet-seed",
        "reversal",
        "trick-room",
        "dragon-dance",
    ]

    {
        move: get_neightbours(move, output_df=False, metric="euclidean")
        for move in lol
    }
    return


@app.cell
def _(moves_df):
    moves_df.columns.tolist()
    return


if __name__ == "__main__":
    app.run()

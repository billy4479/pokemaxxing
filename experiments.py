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
    from sklearn.metrics import silhouette_score
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
    from torch import nn
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, TensorDataset
    from tqdm.auto import tqdm
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
    moves_df = build_moves_table().drop(columns=["effect"])
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
def _(moves_df, stat_change_cols):
    from itertools import product

    _targets = sorted(moves_df["target"].dropna().unique())
    stat_changes_x_target_df = pd.concat(
        [
            ((moves_df["target"] == _target) & (moves_df[_stat_col].fillna(0) > 0))
            .astype(np.float32)
            .rename(
                f"changes_{_stat_col.replace('stat_change_', '')}_for_{_target}_{_direction}"
            )
            for _target, _stat_col, _direction in product(
                _targets, stat_change_cols, ("increase", "decrease")
            )
        ],
        axis=1,
    )
    return (stat_changes_x_target_df,)


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
def _(
    categorical_df,
    flags_df,
    numeric_scaled_df,
    stat_change_df,
    stat_changes_x_target_df,
):
    feature_df = pd.concat(
        [
            numeric_scaled_df,
            stat_change_df,
            categorical_df,
            flags_df,
            stat_changes_x_target_df,
        ],
        axis=1,
    )

    feature_df.shape
    return (feature_df,)


@app.cell
def _(
    categorical_df,
    flags_df,
    numeric_scaled_df,
    stat_change_df,
    stat_changes_x_target_df,
):
    n_numeric = numeric_scaled_df.shape[1] + stat_change_df.shape[1]
    n_categorical = (
        categorical_df.shape[1]
        + flags_df.shape[1]
        + stat_changes_x_target_df.shape[1]
    )

    {"n_numeric": n_numeric, "n_categorical": n_categorical}
    return (n_numeric,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Autoencoder

    The first approach I tried is to train an AE and study its latent space.
    """)
    return


@app.cell
def _():
    seed = 42

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device, seed


@app.cell
def _(n_numeric):
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


    def ae_loss(
        reconstruction_logits,
        inputs,
    ):
        reconstructed_numeric = reconstruction_logits[:, :n_numeric]
        target_numeric = inputs[:, :n_numeric]

        recon_numeric = F.mse_loss(
            reconstructed_numeric,
            target_numeric,
            reduction="mean",
        )

        reconstructed_categorical = reconstruction_logits[:, n_numeric:]
        target_categorical = inputs[:, n_numeric:]
        recon_categorical = F.binary_cross_entropy_with_logits(
            reconstructed_categorical,
            target_categorical,
            reduction="mean",
        )

        total = recon_numeric + recon_categorical

        return total, recon_numeric, recon_categorical

    return MoveAutoencoder, ae_loss


@app.cell
def _(MoveAutoencoder, ae_loss, device):
    def train(dataset: pd.DataFrame, latent_dim: int, epochs: int, learning_rate: float = 1e-4):
        batch_size = 128
    
        x = torch.tensor(dataset.to_numpy(), dtype=torch.float32)
        dataset = TensorDataset(x)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        input_dim = x.shape[1]
        dataset_size = len(dataset)

        model = MoveAutoencoder(input_dim=input_dim, latent_dim=latent_dim).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

        history_rows = []
        epoch_progress = tqdm(range(1, epochs + 1), desc="Training AE", unit="epoch")
        for epoch in epoch_progress:
            model.train()
    
            epoch_total = 0.0
            epoch_numeric = 0.0
            epoch_categorical = 0.0
    
            for (batch,) in dataloader:
                batch = batch.to(device)
    
                optimizer.zero_grad()
                reconstruction_logits_batch, latent_batch = model(batch)
                loss, recon_numeric, recon_categorical = ae_loss(
                    reconstruction_logits_batch,
                    batch,
                )
                loss.backward()
                optimizer.step()
    
                batch_count = batch.shape[0]
                epoch_total += loss.item() * batch_count
                epoch_numeric += recon_numeric.item() * batch_count
                epoch_categorical += recon_categorical.item() * batch_count
    
            history_rows.append(
                {
                    "epoch": epoch,
                    "loss": epoch_total / dataset_size,
                    "recon_numeric": epoch_numeric / dataset_size,
                    "recon_categorical": epoch_categorical / dataset_size,
                }
            )
            epoch_progress.set_postfix(
                loss=f"{history_rows[-1]['loss']:.4f}",
                recon_num=f"{history_rows[-1]['recon_numeric']:.4f}",
                recon_cat=f"{history_rows[-1]['recon_categorical']:.4f}",
            )

        model.eval()
        with torch.no_grad():
            latent_embedding = model.encode(x.to(device))
            reconstruction_logits_eval = model.decode(latent_embedding)
        
        embedding_matrix = latent_embedding.detach().cpu().numpy()


    
        return embedding_matrix, pd.DataFrame(history_rows)

    return (train,)


@app.cell
def _(feature_df, train):
    embedding_matrix, training_history = train(feature_df, 8, 256)
    training_history.tail(10)
    return embedding_matrix, training_history


@app.cell
def _(device, training_history):
    training_summary = {
        "device": str(device),
        "epochs": int(training_history["epoch"].max()),
        "final_loss": float(training_history.iloc[-1]["loss"]),
        "best_loss": float(training_history["loss"].min()),
    }
    training_summary
    return


@app.cell
def _(training_history):
    _fig, _ax = plt.subplots(figsize=(9, 4))

    _ax.plot(training_history["epoch"], training_history["loss"], label="total", linewidth=2)
    _ax.plot(
        training_history["epoch"],
        training_history["recon_numeric"],
        label="recon_numeric",
        linewidth=1.5,
    )
    _ax.plot(
        training_history["epoch"],
        training_history["recon_categorical"],
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
def _(seed):
    def compute_umap_pca(embedding_matrix):
        umap = UMAP(
            random_state=seed,
            n_neighbors=10,
            min_dist=0.75,
            metric="euclidean",
            n_jobs=1,
        ).fit_transform(embedding_matrix)
        pca = PCA(n_components=2).fit_transform(embedding_matrix)
        return (umap.T, pca.T)

    return (compute_umap_pca,)


@app.function
def plot_umap_pca(umap, pca, labels, get_mask, title, legend_title="", legend=True):
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
    s = scatter(ax2, pca)

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

    plot_umap_pca(
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
        for k in range(2, max_k+1):
            kmeans = KMeans(n_clusters=k, random_state=seed, n_init="auto")
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
    kmeans_labels, k = best_kmeans_silhouette(embedding_matrix, max_k=128)
    plot_umap_pca(
        umap,
        pca,
        list(range(k)),
        lambda label: kmeans_labels == label,
        f"2D Embedding colored by KMeans - {k} clusters",
        legend=False
    )
    return k, kmeans_labels


@app.cell
def _(embedding_matrix, kmeans_labels, seed):
    def plot_3d_umap(data, kmeans_labels):
        import plotly.express as px
    
        umap_3 = UMAP(
            n_components=3,
            random_state=seed,
            n_neighbors=10,
            min_dist=0.75,
            metric="euclidean",
            n_jobs=1,
        ).fit_transform(data)
    
        df = pd.DataFrame({
            "UMAP1": umap_3[:, 0],
            "UMAP2": umap_3[:, 1],
            "UMAP3": umap_3[:, 2],
            "cluster": kmeans_labels.astype(str)
        })
    
        fig = px.scatter_3d(
            df,
            x="UMAP1",
            y="UMAP2",
            z="UMAP3",
            color="cluster",
            opacity=0.8,
            title="3D UMAP colored by KMeans labels"
        )
    
        fig.update_traces(marker=dict(size=1.5))
        fig.update_layout(
            showlegend=False,
            scene=dict(
                xaxis_title="UMAP 1",
                yaxis_title="UMAP 2",
                zaxis_title="UMAP 3"
            ),
            width=1000,
            height=1000
        )

        fig.show()

    plot_3d_umap(embedding_matrix, kmeans_labels)
    return


@app.cell
def _(k, kmeans_labels, moves_df):
    [moves_df["move_identifier"][kmeans_labels == i].tolist() for i in range(k)]
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
    move_sample = [
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
        for move in move_sample
    }
    return


@app.cell
def _():
    diagnostic_moves = [
        # Recovery
        "recover",
        "roost",
        "rest",
        "strength-sap",
        # Protection / stalling
        "protect",
        "detect",
        "endure",
        "spiky-shield",
        # Major status
        "toxic",
        "will-o-wisp",
        "thunder-wave",
        "spore",
        "sleep-powder",
        "glare",
        # Setup sweepers
        "swords-dance",
        "nasty-plot",
        "dragon-dance",
        "quiver-dance",
        "shell-smash",
        "calm-mind",
        "bulk-up",
        "agility",
        # Target stat manipulation / disruption
        "swagger",
        "flatter",
        "charm",
        "fake-tears",
        "eerie-impulse",
        "parting-shot",
        # Pivoting / switching
        "u-turn",
        "volt-switch",
        "flip-turn",
        "baton-pass",
        "teleport",
        "chilly-reception",
        # Multi-hit
        "bullet-seed",
        "icicle-spear",
        "rock-blast",
        "pin-missile",
        "arm-thrust",
        "population-bomb",
        # Conditional / variable power
        "flail",
        "reversal",
        "low-kick",
        "grass-knot",
        "gyro-ball",
        "electro-ball",
        "stored-power",
        "power-trip",
        # Trapping / residual
        "sand-tomb",
        "whirlpool",
        "fire-spin",
        "infestation",
        "magma-storm",
        "thunder-cage",
        # Field / room / terrain / weather
        "trick-room",
        "wonder-room",
        "magic-room",
        "gravity",
        "tailwind",
        "sunny-day",
        "rain-dance",
        "electric-terrain",
        "misty-terrain",
        # Hazards / screens
        "stealth-rock",
        "spikes",
        "toxic-spikes",
        "sticky-web",
        "reflect",
        "light-screen",
        "aurora-veil",
        # Priority
        "quick-attack",
        "extreme-speed",
        "mach-punch",
        "sucker-punch",
        "fake-out",
        "protect",
        "helping-hand",
        "trick-room",
        # Drain / recoil
        "drain-punch",
        "giga-drain",
        "horn-leech",
        "brave-bird",
        "flare-blitz",
        "wild-charge",
        "head-smash",
        # High-power drawback / commitment
        "hyper-beam",
        "giga-impact",
        "blast-burn",
        "frenzy-plant",
        "hydro-cannon",
        "steel-beam",
        "leaf-storm",
        "draco-meteor",
        # Weird control / unique effects
        "taunt",
        "encore",
        "disable",
        "substitute",
        "perish-song",
        "destiny-bond",
        "trick",
        "switcheroo",
        "topsy-turvy",
        "haze",
    ]
    return (diagnostic_moves,)


@app.cell
def _(diagnostic_moves, get_neightbours):
    {
        move: get_neightbours(move, output_df=False, metric="euclidean", count=5)
        for move in diagnostic_moves
    }
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

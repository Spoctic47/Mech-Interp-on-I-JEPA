"""
src/analysis/visualization.py
-------------------------------
Publication-quality visualization utilities for the I-JEPA probing experiment.

Plots produced:
  1. Layer accuracy curve (train + val)
  2. CKA heatmap (L x L)
  3. t-SNE projection at selected layers
  4. Summary dashboard (all plots in one figure)
"""

import os
import logging
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Use non-interactive backend to avoid display issues on headless machines
matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Aesthetic settings
# ---------------------------------------------------------------------------

PALETTE = {
    "val_acc":   "#4C9BE8",   # soft blue
    "train_acc": "#A8D8A8",   # soft green
    "cka_cmap":  "magma",
    "bg":        "#0F1117",
    "fg":        "#E8E8E8",
    "grid":      "#2A2D38",
}

def _apply_dark_style():
    plt.rcParams.update({
        "figure.facecolor":  PALETTE["bg"],
        "axes.facecolor":    PALETTE["bg"],
        "axes.edgecolor":    PALETTE["grid"],
        "axes.labelcolor":   PALETTE["fg"],
        "xtick.color":       PALETTE["fg"],
        "ytick.color":       PALETTE["fg"],
        "text.color":        PALETTE["fg"],
        "grid.color":        PALETTE["grid"],
        "grid.linestyle":    "--",
        "grid.alpha":        0.4,
        "font.family":       "DejaVu Sans",
        "font.size":         11,
        "axes.titlesize":    13,
        "axes.titleweight":  "bold",
        "legend.facecolor":  "#1A1D24",
        "legend.edgecolor":  PALETTE["grid"],
    })


# ---------------------------------------------------------------------------
# 1. Layer Accuracy Curve
# ---------------------------------------------------------------------------

def plot_layer_accuracy_curve(
    results: List[Dict],
    save_path: str,
    title: str = "I-JEPA Layer-wise Linear Probing Accuracy",
):
    """
    Plot val and train accuracy across all probed layers.

    Args:
        results:    List of dicts with keys: layer, val_acc, train_acc.
        save_path:  Where to save the PNG.
        title:      Plot title.
    """
    _apply_dark_style()
    layers    = [r["layer"]     for r in results]
    val_accs  = [r["val_acc"]   for r in results]
    train_accs= [r["train_acc"] for r in results]

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(layers, val_accs,   color=PALETTE["val_acc"],   linewidth=2.5,
            marker="o", markersize=5, label="Val Accuracy")
    ax.fill_between(layers, val_accs, alpha=0.15, color=PALETTE["val_acc"])

    ax.plot(layers, train_accs, color=PALETTE["train_acc"], linewidth=2,
            linestyle="--", marker="s", markersize=4, label="Train Accuracy")

    # Annotate the peak val layer
    best_layer = layers[int(np.argmax(val_accs))]
    best_acc   = max(val_accs)
    ax.annotate(
        f"Peak: Layer {best_layer}\n{best_acc:.2%}",
        xy=(best_layer, best_acc),
        xytext=(best_layer + 0.5, best_acc - 0.05),
        fontsize=9,
        color=PALETTE["fg"],
        arrowprops=dict(arrowstyle="->", color=PALETTE["fg"], lw=1.2),
    )

    ax.set_xlabel("Transformer Block (Layer Index)")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.set_xticks(layers)
    ax.set_xticklabels([str(l) for l in layers], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.legend(loc="lower right")
    ax.grid(True)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved accuracy curve → {save_path}")


# ---------------------------------------------------------------------------
# 2. CKA Heatmap
# ---------------------------------------------------------------------------

def plot_cka_heatmap(
    cka_matrix: np.ndarray,
    layer_indices: List[int],
    save_path: str,
    title: str = "Linear CKA: Layer Representational Similarity",
):
    """
    Plot the L x L CKA similarity matrix as a heatmap.

    Args:
        cka_matrix:   (L, L) array with CKA values in [0, 1].
        layer_indices: Layer index labels for axes.
        save_path:    Where to save the PNG.
        title:        Plot title.
    """
    _apply_dark_style()
    fig, ax = plt.subplots(figsize=(9, 8))

    mask = np.zeros_like(cka_matrix, dtype=bool)  # no masking — show full matrix

    sns.heatmap(
        cka_matrix,
        ax=ax,
        cmap=PALETTE["cka_cmap"],
        vmin=0, vmax=1,
        xticklabels=layer_indices,
        yticklabels=layer_indices,
        linewidths=0,
        annot=False,
        cbar_kws={"label": "CKA Similarity", "shrink": 0.8},
    )

    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Layer Index")
    ax.set_title(title)
    ax.tick_params(axis="both", labelsize=8)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved CKA heatmap → {save_path}")


# ---------------------------------------------------------------------------
# 3. t-SNE Visualization
# ---------------------------------------------------------------------------

def plot_tsne(
    features: Dict[int, np.ndarray],
    labels: np.ndarray,
    layer_indices: List[int],
    class_names: List[str],
    save_path: str,
    max_samples: int = 300,
    perplexity: float = 30.0,
    seed: int = 42,
):
    """
    Plot t-SNE of feature embeddings at selected layers.

    Args:
        features:     Dict layer_idx -> (N, D) array.
        labels:       (N,) class label array.
        layer_indices: Which layers to visualize.
        class_names:  Human-readable class names.
        save_path:    Where to save the PNG.
        max_samples:  Subsample for speed.
        perplexity:   t-SNE perplexity.
        seed:         RNG seed.
    """
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA

    _apply_dark_style()

    n_plots = len(layer_indices)
    ncols   = min(n_plots, 3)
    nrows   = (n_plots + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = np.array(axes).flatten()

    n = features[layer_indices[0]].shape[0]
    if n > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, max_samples, replace=False)
    else:
        idx = np.arange(n)

    num_classes = len(class_names)
    try:
        cmap = matplotlib.colormaps["tab10"]
    except (AttributeError, KeyError):
        cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(num_classes)]

    for ax_i, (layer_idx, ax) in enumerate(zip(layer_indices, axes)):
        X = features[layer_idx][idx]
        y = labels[idx]

        # Reduce to 50 dims with PCA first (speeds up t-SNE)
        pca_dim = min(50, X.shape[1], X.shape[0])
        if X.shape[1] > pca_dim:
            pca = PCA(n_components=pca_dim, random_state=seed)
            X   = pca.fit_transform(X)

        try:
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=seed, max_iter=500)
        except TypeError:
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=seed, n_iter=500)
        X_2d = tsne.fit_transform(X)

        for cls_i, cls_name in enumerate(class_names):
            mask = y == cls_i
            ax.scatter(
                X_2d[mask, 0], X_2d[mask, 1],
                c=[colors[cls_i]],
                label=cls_name,
                alpha=0.75,
                s=18,
                edgecolors="none",
            )

        ax.set_title(f"Layer {layer_idx}")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_facecolor(PALETTE["bg"])

        if ax_i == 0:
            ax.legend(fontsize=7, markerscale=1.5, loc="upper right",
                      ncol=2, framealpha=0.5)

    # Hide unused axes
    for ax in axes[n_plots:]:
        ax.set_visible(False)

    fig.suptitle("t-SNE of I-JEPA Representations by Layer", fontsize=14, y=1.01)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved t-SNE plots → {save_path}")


# ---------------------------------------------------------------------------
# 4. Summary Dashboard
# ---------------------------------------------------------------------------

def plot_summary_dashboard(
    results:       List[Dict],
    cka_matrix:    Optional[np.ndarray],
    layer_indices: List[int],
    save_path:     str,
):
    """
    Single-figure summary: accuracy curve + CKA heatmap side-by-side.
    """
    _apply_dark_style()

    if cka_matrix is not None:
        fig = plt.figure(figsize=(18, 6))
        gs  = gridspec.GridSpec(1, 2, width_ratios=[2, 1.2], wspace=0.35)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
    else:
        fig, ax1 = plt.subplots(figsize=(12, 5))
        ax2 = None

    # --- Accuracy curve ---
    layers     = [r["layer"]     for r in results]
    val_accs   = [r["val_acc"]   for r in results]
    train_accs = [r["train_acc"] for r in results]

    ax1.plot(layers, val_accs,   color=PALETTE["val_acc"],   linewidth=2.5,
             marker="o", markersize=5, label="Val Accuracy")
    ax1.fill_between(layers, val_accs, alpha=0.12, color=PALETTE["val_acc"])
    ax1.plot(layers, train_accs, color=PALETTE["train_acc"], linewidth=2,
             linestyle="--", marker="s", markersize=4, label="Train Accuracy")

    best_idx  = int(np.argmax(val_accs))
    ax1.axvline(x=layers[best_idx], color="#FF9B54", linestyle=":", linewidth=1.5,
                label=f"Best Layer {layers[best_idx]}")

    ax1.set_xlabel("Layer Index")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Layer-wise Linear Probing Accuracy")
    ax1.set_ylim(0, 1.05)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax1.legend(loc="lower right")
    ax1.grid(True)

    # --- CKA heatmap ---
    if ax2 is not None and cka_matrix is not None:
        # Only label every 4th layer to reduce clutter
        tick_labels = [str(l) if l % 4 == 0 else "" for l in layer_indices]
        sns.heatmap(
            cka_matrix,
            ax=ax2,
            cmap=PALETTE["cka_cmap"],
            vmin=0, vmax=1,
            xticklabels=tick_labels,
            yticklabels=tick_labels,
            linewidths=0,
            cbar_kws={"label": "CKA", "shrink": 0.8},
        )
        ax2.set_title("Layer-to-Layer CKA Similarity")
        ax2.set_xlabel("Layer Index")
        ax2.set_ylabel("Layer Index")
        ax2.tick_params(axis="both", labelsize=8)

    fig.suptitle("I-JEPA ViT-Huge — Mechanistic Probing Results", fontsize=15, y=1.02)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved summary dashboard → {save_path}")

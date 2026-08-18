"""
run_probing.py
--------------
Main entrypoint for the I-JEPA layer-wise linear probing experiment.

Usage:
    python run_probing.py                               # default config
    python run_probing.py --config configs/probing_config.yaml
    python run_probing.py --dataset cifar10 --num_classes 10
    python run_probing.py --dataset tiny_imagenet --num_classes 20
    python run_probing.py --reuse_features              # skip extraction if cached

What this script does:
  1. Load config + set reproducibility seed.
  2. Build the I-JEPA ViT-Huge encoder (frozen).
  3. Build train / val dataloaders.
  4. Extract layer-wise features (single forward pass per batch).
     - Optionally cache features to disk for fast re-runs.
  5. Train one Logistic Regression probe per layer.
  6. (Optional) Compute CKA similarity matrix between layers.
  7. Generate plots: accuracy curve, CKA heatmap, t-SNE, summary dashboard.
  8. Save results CSV.
"""

import os
import sys
import time
import random
import logging
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

# Silence some noisy warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ── Local imports ────────────────────────────────────────────────────────────
from src.models.ijepa_builder import build_ijepa_encoder
from src.data.imagenet_loader import build_dataloaders
from src.experiments.probing import (
    LayerwiseFeatureExtractor,
    run_probing_experiment,
    save_features,
    load_features,
    features_cached,
)
from src.analysis.cka import compute_cka_matrix
from src.analysis.visualization import (
    plot_layer_accuracy_curve,
    plot_cka_heatmap,
    plot_tsne,
    plot_summary_dashboard,
)

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_probing")


# ── Reproducibility ──────────────────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ── Config helpers ───────────────────────────────────────────────────────────
def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def merge_cli_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    """Allow CLI flags to override YAML config values."""
    if args.dataset:
        cfg["data"]["dataset"] = args.dataset
    if args.num_classes is not None:
        cfg["data"]["num_classes"] = args.num_classes
    if args.images_per_class is not None:
        cfg["data"]["images_per_class"] = args.images_per_class
    if args.batch_size is not None:
        cfg["data"]["batch_size"] = args.batch_size  # not in YAML by default; added here
    return cfg


# ── Argument parsing ─────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="I-JEPA Layer-wise Linear Probing Experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str,
        default="configs/probing_config.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--dataset", type=str,
        choices=["cifar10", "tiny_imagenet", "imagenet_hf"],
        default=None,
        help="Override dataset choice from config.",
    )
    parser.add_argument(
        "--num_classes", type=int, default=None,
        help="Number of classes to include in probing.",
    )
    parser.add_argument(
        "--images_per_class", type=int, default=None,
        help="Max images per class (train + val combined).",
    )
    parser.add_argument(
        "--batch_size", type=int, default=None,
        help="DataLoader batch size (default: 64 for GPU, 32 for CPU).",
    )
    parser.add_argument(
        "--reuse_features", action="store_true",
        help="Load cached features from disk (skip extraction if they exist).",
    )
    parser.add_argument(
        "--no_cka", action="store_true",
        help="Skip CKA computation (faster run).",
    )
    parser.add_argument(
        "--no_tsne", action="store_true",
        help="Skip t-SNE visualization (faster run).",
    )
    return parser.parse_args()


# ── Device selection ─────────────────────────────────────────────────────────
def get_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        logger.info(f"Using GPU: {name}")
    else:
        dev = torch.device("cpu")
        logger.warning(
            "No GPU detected — running on CPU. "
            "Feature extraction may take 20–30 minutes for ViT-Huge."
        )
    return dev


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    t_start = time.perf_counter()
    args    = parse_args()

    # 1. Config
    logger.info(f"Loading config from {args.config}")
    cfg = load_config(args.config)
    cfg = merge_cli_overrides(cfg, args)

    seed = cfg["experiment"].get("seed", 42)
    set_seed(seed)
    logger.info(f"Seed set to {seed}")

    # Resolve batch size: default depends on device
    device     = get_device()
    batch_size = cfg.get("data", {}).get("batch_size", 64 if device.type == "cuda" else 32)

    out_dir      = cfg["output"]["save_dir"]
    features_dir = cfg["output"]["features_dir"]
    os.makedirs(out_dir, exist_ok=True)

    # 2. Model
    encoder = build_ijepa_encoder(device)
    num_layers = len(encoder.blocks)

    # 3. Determine which layers to probe
    probe_layers_cfg = cfg["model"].get("probe_layers", "all")
    if probe_layers_cfg == "all":
        layer_indices = list(range(num_layers))
    else:
        layer_indices = list(probe_layers_cfg)

    # 4. Data
    data_cfg = cfg["data"]
    logger.info(
        f"Dataset: {data_cfg['dataset']} | "
        f"Classes: {data_cfg['num_classes']} | "
        f"Images/class: {data_cfg['images_per_class']}"
    )

    train_loader, val_loader, class_names, num_classes = build_dataloaders(
        dataset          = data_cfg["dataset"],
        data_root        = data_cfg.get("data_root", "./data"),
        image_size       = data_cfg.get("image_size", 224),
        num_classes      = data_cfg["num_classes"],
        images_per_class = data_cfg["images_per_class"],
        val_fraction     = cfg["probe"].get("val_fraction", 0.2),
        batch_size       = batch_size,
        num_workers      = data_cfg.get("num_workers", 2),
        seed             = seed,
        hf_dataset_name  = data_cfg.get("hf_dataset_name", "imagenet-1k"),
    )
    logger.info(f"Classes: {class_names}")

    # 5. Feature extraction
    extractor = LayerwiseFeatureExtractor(encoder, layer_indices=layer_indices)
    pooling   = cfg["probe"].get("pooling", "mean")

    def _extract_split(loader, split_name):
        if args.reuse_features and features_cached(features_dir, split_name, layer_indices):
            logger.info(f"Loading cached {split_name} features ...")
            return load_features(features_dir, split_name, layer_indices)
        else:
            logger.info(f"Extracting {split_name} features ...")
            feats, lbls = extractor.extract(loader, device, pooling=pooling)
            if cfg["output"].get("save_features", True):
                save_features(feats, lbls, features_dir, split_name)
            return feats, lbls

    train_features, train_labels = _extract_split(train_loader, "train")
    val_features,   val_labels   = _extract_split(val_loader,   "val")

    # 6. Probing
    probe_cfg = cfg["probe"]
    results = run_probing_experiment(
        train_features = train_features,
        train_labels   = train_labels,
        val_features   = val_features,
        val_labels     = val_labels,
        layer_indices  = layer_indices,
        C              = probe_cfg.get("C", 1.0),
        max_iter       = probe_cfg.get("max_iter", 1000),
        solver         = probe_cfg.get("solver", "lbfgs"),
        seed           = seed,
    )

    # Save CSV
    df = pd.DataFrame(results)
    csv_path = os.path.join(out_dir, "probe_results.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"Results saved → {csv_path}")

    best = df.loc[df["val_acc"].idxmax()]
    logger.info(
        f"\n{'='*55}\n"
        f"  Best layer : {int(best['layer'])}\n"
        f"  Val  acc   : {best['val_acc']:.4f} ({best['val_acc']:.1%})\n"
        f"  Train acc  : {best['train_acc']:.4f} ({best['train_acc']:.1%})\n"
        f"{'='*55}"
    )

    # 7. CKA
    cka_matrix = None
    analysis_cfg = cfg.get("analysis", {})
    if analysis_cfg.get("compute_cka", True) and not args.no_cka:
        # Use train features for CKA (larger set)
        cka_matrix = compute_cka_matrix(train_features, layer_indices)
        np.save(os.path.join(out_dir, "cka_matrix.npy"), cka_matrix)

    # 8. Visualizations
    plot_layer_accuracy_curve(
        results   = results,
        save_path = os.path.join(out_dir, "layer_accuracy_curve.png"),
    )

    if cka_matrix is not None:
        plot_cka_heatmap(
            cka_matrix    = cka_matrix,
            layer_indices = layer_indices,
            save_path     = os.path.join(out_dir, "cka_heatmap.png"),
        )

    plot_summary_dashboard(
        results       = results,
        cka_matrix    = cka_matrix,
        layer_indices = layer_indices,
        save_path     = os.path.join(out_dir, "summary_dashboard.png"),
    )

    if not args.no_tsne:
        tsne_layers = analysis_cfg.get("tsne_layers", [0, 7, 15, 23, 31])
        # Filter to only layers that were actually probed
        tsne_layers = [l for l in tsne_layers if l in layer_indices]
        if tsne_layers:
            plot_tsne(
                features      = train_features,
                labels        = train_labels,
                layer_indices = tsne_layers,
                class_names   = class_names,
                save_path     = os.path.join(out_dir, "tsne_visualization.png"),
                max_samples   = 300,
                seed          = seed,
            )

    elapsed = time.perf_counter() - t_start
    logger.info(
        f"\nExperiment complete in {elapsed/60:.1f} min.\n"
        f"All outputs saved to: {os.path.abspath(out_dir)}\n"
        f"  ├── probe_results.csv\n"
        f"  ├── layer_accuracy_curve.png\n"
        f"  ├── cka_heatmap.png\n"
        f"  ├── summary_dashboard.png\n"
        f"  └── tsne_visualization.png\n"
    )


if __name__ == "__main__":
    main()

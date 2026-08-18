"""
src/experiments/probing.py
---------------------------
Layer-wise linear probing on I-JEPA ViT-Huge.

Pipeline:
  1. Register forward hooks on every transformer block.
  2. Run a single forward pass per batch → captures all 32 layer outputs at once.
  3. Aggregate (N, seq_len, dim) → (N, dim) via mean-pooling.
  4. After iterating the full DataLoader, train a Logistic Regression probe
     for each layer independently.
  5. Return per-layer accuracy, probe objects, and raw features (optionally saved).

Design choices:
  - Single-pass feature extraction: all layers captured in one forward pass
    (efficient — no repeated forward passes).
  - CPU-side accumulation: features moved off GPU immediately to avoid OOM
    on large ViT-Huge (embed_dim=1280, 32 layers × 500 images × 1280 = ~330MB).
  - Reproducibility: sklearn probes seeded; numpy RNG fixed.
"""

import os
import time
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature Extraction
# ---------------------------------------------------------------------------

class LayerwiseFeatureExtractor:
    """
    Extracts mean-pooled CLS/patch token representations from every
    transformer block in a single forward pass.

    Usage:
        extractor = LayerwiseFeatureExtractor(model)
        features  = extractor.extract(dataloader, device)
        # features: dict[layer_idx] -> np.ndarray of shape (N, embed_dim)
    """

    def __init__(self, model: nn.Module, layer_indices: Optional[List[int]] = None):
        """
        Args:
            model:         The I-JEPA ViT encoder (timm model).
            layer_indices: Which block indices to probe. None = all blocks.
        """
        self.model = model
        self.blocks = list(model.blocks)
        self.num_layers = len(self.blocks)

        if layer_indices is None:
            self.layer_indices = list(range(self.num_layers))
        else:
            self.layer_indices = [i for i in layer_indices if 0 <= i < self.num_layers]

        logger.info(f"Probing {len(self.layer_indices)} layers out of {self.num_layers} total.")

    @torch.no_grad()
    def extract(
        self,
        dataloader: DataLoader,
        device: torch.device,
        pooling: str = "mean",
    ) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
        """
        Run feature extraction over an entire DataLoader.

        Args:
            dataloader: Yields (images, labels) batches.
            device:     Computation device.
            pooling:    "mean" to average patch tokens; "cls" for first token only.

        Returns:
            features: dict mapping layer_idx -> np.ndarray (N, D)
            labels:   np.ndarray of shape (N,)
        """
        self.model.eval()
        self.model.to(device)

        # Accumulators — kept on CPU to avoid GPU OOM
        layer_feats: Dict[int, List[np.ndarray]] = {i: [] for i in self.layer_indices}
        all_labels: List[np.ndarray] = []

        # Temporary buffer to hold the hook outputs for the current batch
        _hook_buffer: Dict[int, torch.Tensor] = {}

        def make_hook(layer_idx: int):
            def hook(module, inp, out):
                # out: (B, seq_len, D)
                if pooling == "mean":
                    pooled = out.mean(dim=1)           # (B, D)
                elif pooling == "cls":
                    pooled = out[:, 0, :]              # (B, D) — CLS token
                else:
                    raise ValueError(f"Unknown pooling: {pooling}")
                # Move to CPU immediately
                _hook_buffer[layer_idx] = pooled.cpu()
            return hook

        # Register hooks
        handles = []
        for idx in self.layer_indices:
            h = self.blocks[idx].register_forward_hook(make_hook(idx))
            handles.append(h)

        try:
            for images, labels in tqdm(dataloader, desc="Extracting features", unit="batch"):
                images = images.to(device, non_blocking=True)
                _hook_buffer.clear()

                _ = self.model(images)  # single forward pass fills _hook_buffer

                for idx in self.layer_indices:
                    layer_feats[idx].append(_hook_buffer[idx].numpy())  # (B, D)

                all_labels.append(labels.numpy())

        finally:
            for h in handles:
                h.remove()

        # Stack into arrays
        features = {idx: np.concatenate(layer_feats[idx], axis=0) for idx in self.layer_indices}
        labels   = np.concatenate(all_labels, axis=0)

        logger.info(
            f"Extracted features: {len(labels)} samples, "
            f"{len(features)} layers, embed_dim={next(iter(features.values())).shape[1]}"
        )
        return features, labels


# ---------------------------------------------------------------------------
# Feature caching
# ---------------------------------------------------------------------------

def save_features(features: Dict[int, np.ndarray], labels: np.ndarray, save_dir: str, split: str):
    """Save extracted features and labels to .npy files."""
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, f"{split}_labels.npy"), labels)
    for layer_idx, feats in features.items():
        np.save(os.path.join(save_dir, f"{split}_layer_{layer_idx:02d}.npy"), feats)
    logger.info(f"Features saved to {save_dir}/")


def load_features(save_dir: str, split: str, layer_indices: List[int]) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
    """Load pre-extracted features from disk."""
    labels = np.load(os.path.join(save_dir, f"{split}_labels.npy"))
    features = {}
    for idx in layer_indices:
        path = os.path.join(save_dir, f"{split}_layer_{idx:02d}.npy")
        features[idx] = np.load(path)
    logger.info(f"Loaded features from {save_dir}/ ({len(features)} layers)")
    return features, labels


def features_cached(save_dir: str, split: str, layer_indices: List[int]) -> bool:
    """Check whether cached features exist for all requested layers."""
    label_path = os.path.join(save_dir, f"{split}_labels.npy")
    if not os.path.exists(label_path):
        return False
    for idx in layer_indices:
        if not os.path.exists(os.path.join(save_dir, f"{split}_layer_{idx:02d}.npy")):
            return False
    return True


# ---------------------------------------------------------------------------
# Probe Training
# ---------------------------------------------------------------------------

def train_single_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    C: float = 1.0,
    max_iter: int = 1000,
    solver: str = "lbfgs",
    seed: int = 42,
) -> Tuple[float, object]:
    """
    Train a Logistic Regression probe on features from a single layer.

    Returns:
        (val_accuracy, trained_pipeline)
    """
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=C,
            max_iter=max_iter,
            solver=solver,
            random_state=seed,
            n_jobs=-1,
        )
    )
    clf.fit(X_train, y_train)
    preds = clf.predict(X_val)
    acc = accuracy_score(y_val, preds)
    return acc, clf


def run_probing_experiment(
    train_features: Dict[int, np.ndarray],
    train_labels:   np.ndarray,
    val_features:   Dict[int, np.ndarray],
    val_labels:     np.ndarray,
    layer_indices:  List[int],
    C:              float = 1.0,
    max_iter:       int   = 1000,
    solver:         str   = "lbfgs",
    seed:           int   = 42,
) -> List[Dict]:
    """
    Train and evaluate one linear probe per layer.

    Returns:
        List of result dicts with keys:
          layer, train_acc, val_acc, train_time_s
    """
    results = []
    logger.info(f"Training {len(layer_indices)} probes ...")

    for layer_idx in tqdm(layer_indices, desc="Probing layers"):
        X_tr = train_features[layer_idx]
        X_va = val_features[layer_idx]

        t0 = time.perf_counter()
        val_acc, clf = train_single_probe(
            X_tr, train_labels,
            X_va, val_labels,
            C=C, max_iter=max_iter, solver=solver, seed=seed,
        )
        elapsed = time.perf_counter() - t0

        # Also compute train accuracy to check for underfitting
        train_acc = accuracy_score(train_labels, clf.predict(X_tr))

        results.append({
            "layer":        layer_idx,
            "val_acc":      round(val_acc,   4),
            "train_acc":    round(train_acc, 4),
            "train_time_s": round(elapsed,   3),
        })
        logger.info(
            f"  Layer {layer_idx:2d} | val_acc={val_acc:.4f} | "
            f"train_acc={train_acc:.4f} | time={elapsed:.2f}s"
        )

    return results

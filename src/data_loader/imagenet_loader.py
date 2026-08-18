"""
src/data/imagenet_loader.py
----------------------------
Flexible dataset loader for the I-JEPA probing experiment.

Supports:
  - "cifar10"      : CIFAR-10 via torchvision (auto-download, no token)
  - "tiny_imagenet": Tiny-ImageNet-200 (auto-download, no token)
  - "imagenet_hf"  : ImageNet-1k subset via HuggingFace datasets (needs HF token)

Returns:
  train_loader, val_loader, class_names, num_classes
"""

import os
import random
import logging
from pathlib import Path
from typing import Tuple, List, Optional

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Standard ImageNet normalization (used for all datasets since model is I-JEPA
# trained on ImageNet — even for CIFAR-10, normalizing to ImageNet stats works
# well because we're evaluating transfer, not training the encoder).
# ---------------------------------------------------------------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def get_transforms(image_size: int = 224) -> transforms.Compose:
    """Standard evaluation transforms used during probing (no augmentation)."""
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# Dataset builders
# ---------------------------------------------------------------------------

def _build_cifar10(data_root: str, image_size: int):
    """Loads CIFAR-10 train and val (test) splits."""
    transform = get_transforms(image_size)
    train_ds = datasets.CIFAR10(root=data_root, train=True,  download=True, transform=transform)
    val_ds   = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)
    class_names = train_ds.classes
    return train_ds, val_ds, class_names


def _build_tiny_imagenet(data_root: str, image_size: int):
    """
    Loads Tiny-ImageNet-200.
    Downloads and extracts the dataset if not present.
    """
    import zipfile
    import urllib.request

    root = Path(data_root) / "tiny-imagenet-200"
    if not root.exists():
        url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
        zip_path = Path(data_root) / "tiny-imagenet-200.zip"
        logger.info(f"Downloading Tiny-ImageNet from {url} ...")
        urllib.request.urlretrieve(url, zip_path)
        logger.info("Extracting Tiny-ImageNet ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(data_root)
        zip_path.unlink()

    # Fix Tiny-ImageNet val folder structure for ImageFolder compatibility
    _fix_tiny_imagenet_val_structure(root / "val")

    transform = get_transforms(image_size)
    train_ds  = datasets.ImageFolder(str(root / "train"), transform=transform)
    val_ds    = datasets.ImageFolder(str(root / "val"),   transform=transform)
    class_names = train_ds.classes
    return train_ds, val_ds, class_names


def _fix_tiny_imagenet_val_structure(val_dir: Path):
    """
    Tiny-ImageNet val images are all flat in val/images/.
    This reorganises them into val/<class>/ subfolders so ImageFolder works.
    Only runs once (checks for a sentinel file).
    """
    sentinel = val_dir / ".restructured"
    if sentinel.exists():
        return

    annotations = val_dir / "val_annotations.txt"
    if not annotations.exists():
        logger.warning("val_annotations.txt not found — skipping val restructure.")
        return

    images_dir = val_dir / "images"
    with open(annotations) as f:
        for line in f:
            parts = line.strip().split("\t")
            img_name, class_id = parts[0], parts[1]
            class_dir = val_dir / class_id
            class_dir.mkdir(exist_ok=True)
            src = images_dir / img_name
            dst = class_dir / img_name
            if src.exists() and not dst.exists():
                src.rename(dst)

    sentinel.touch()
    logger.info("Tiny-ImageNet val folder restructured.")


def _build_imagenet_hf(data_root: str, image_size: int, hf_dataset_name: str):
    """
    Loads an ImageNet-1k subset via HuggingFace datasets.
    Requires `pip install datasets` and a valid HF token.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "HuggingFace `datasets` package not found. "
            "Install it with: pip install datasets"
        )

    transform = get_transforms(image_size)

    class HFDatasetWrapper(torch.utils.data.Dataset):
        def __init__(self, hf_split, transform):
            self.ds = hf_split
            self.transform = transform

        def __len__(self):
            return len(self.ds)

        def __getitem__(self, idx):
            item = self.ds[idx]
            img = item["image"]
            if img.mode != "RGB":
                img = img.convert("RGB")
            label = item["label"]
            return self.transform(img), label

    logger.info(f"Loading {hf_dataset_name} from HuggingFace (streaming=False) ...")
    raw = load_dataset(hf_dataset_name, trust_remote_code=True)
    train_ds = HFDatasetWrapper(raw["train"],      transform)
    val_ds   = HFDatasetWrapper(raw["validation"], transform)
    # Retrieve class names from the label feature
    class_names = raw["train"].features["label"].names
    return train_ds, val_ds, class_names


# ---------------------------------------------------------------------------
# Subset helpers
# ---------------------------------------------------------------------------

def _class_balanced_subset(
    dataset,
    num_classes: int,
    images_per_class: int,
    seed: int = 42,
) -> Subset:
    """
    Returns a Subset with exactly `images_per_class` images from each of
    the first `num_classes` classes, sampled randomly.
    """
    rng = random.Random(seed)

    # Build a mapping: class_idx -> list of dataset indices
    class_to_indices = {}
    for idx, (_, label) in enumerate(dataset):
        if label not in class_to_indices:
            class_to_indices[label] = []
        class_to_indices[label].append(idx)

    selected = []
    for cls in sorted(class_to_indices.keys())[:num_classes]:
        pool = class_to_indices[cls]
        n    = min(images_per_class, len(pool))
        selected.extend(rng.sample(pool, n))

    return Subset(dataset, selected)


def _fast_class_balanced_subset(
    dataset,
    num_classes: int,
    images_per_class: int,
    seed: int = 42,
) -> Subset:
    """
    Faster version that accesses .targets / .labels attribute when available
    (avoids iterating through the whole dataset).
    """
    rng = random.Random(seed)

    # Try to get targets without loading images
    if hasattr(dataset, "targets"):
        targets = dataset.targets
    elif hasattr(dataset, "labels"):
        targets = dataset.labels
    else:
        # Fall back to slow path
        return _class_balanced_subset(dataset, num_classes, images_per_class, seed)

    class_to_indices = {}
    for idx, label in enumerate(targets):
        label = int(label)
        if label not in class_to_indices:
            class_to_indices[label] = []
        class_to_indices[label].append(idx)

    selected = []
    for cls in sorted(class_to_indices.keys())[:num_classes]:
        pool = class_to_indices[cls]
        n    = min(images_per_class, len(pool))
        selected.extend(rng.sample(pool, n))

    return Subset(dataset, selected)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_dataloaders(
    dataset: str = "cifar10",
    data_root: str = "./data",
    image_size: int = 224,
    num_classes: int = 10,
    images_per_class: int = 50,
    val_fraction: float = 0.2,
    batch_size: int = 64,
    num_workers: int = 2,
    seed: int = 42,
    hf_dataset_name: str = "imagenet-1k",
) -> Tuple[DataLoader, DataLoader, List[str], int]:
    """
    Build train and val DataLoaders for the probing experiment.

    Args:
        dataset:           "cifar10" | "tiny_imagenet" | "imagenet_hf"
        data_root:         Local path for downloads.
        image_size:        Target image resolution (default 224 for ViT-Huge).
        num_classes:       How many classes to include.
        images_per_class:  Max images per class (train + val combined).
        val_fraction:      Fraction of selected images held out for evaluation.
        batch_size:        DataLoader batch size.
        num_workers:       DataLoader workers (use 0 on Windows if issues).
        seed:              RNG seed for reproducibility.
        hf_dataset_name:   HuggingFace dataset name (only used for imagenet_hf).

    Returns:
        (train_loader, val_loader, class_names, num_classes)
    """
    os.makedirs(data_root, exist_ok=True)
    dataset = dataset.lower()

    logger.info(f"Building dataset: {dataset} | classes={num_classes} | img_per_class={images_per_class}")

    # 1. Load full datasets
    if dataset == "cifar10":
        train_full, val_full, class_names = _build_cifar10(data_root, image_size)
    elif dataset == "tiny_imagenet":
        train_full, val_full, class_names = _build_tiny_imagenet(data_root, image_size)
    elif dataset == "imagenet_hf":
        train_full, val_full, class_names = _build_imagenet_hf(data_root, image_size, hf_dataset_name)
    else:
        raise ValueError(f"Unknown dataset: '{dataset}'. Choose from: cifar10, tiny_imagenet, imagenet_hf")

    # Clamp num_classes to what's available
    num_classes = min(num_classes, len(class_names))
    class_names = class_names[:num_classes]

    # 2. Create class-balanced subsets
    train_n  = max(1, int(images_per_class * (1 - val_fraction)))
    val_n    = max(1, int(images_per_class * val_fraction))

    train_subset = _fast_class_balanced_subset(train_full, num_classes, train_n, seed=seed)
    val_subset   = _fast_class_balanced_subset(val_full,   num_classes, val_n,   seed=seed + 1)

    logger.info(f"Train samples: {len(train_subset)} | Val samples: {len(val_subset)}")

    # 3. Build DataLoaders — use pin_memory only when GPU is available
    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )

    return train_loader, val_loader, class_names, num_classes

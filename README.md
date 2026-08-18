# I-JEPA Mechanistic Interpretability — Layer-wise Probing

This repository contains a well-structured **linear probing** experiment on the I-JEPA ViT-Huge encoder.
The goal is to understand what features/concepts each transformer block encodes by training a lightweight Logistic Regression probe at every layer and tracking how accuracy evolves with depth.

## What the experiment does

1. **Loads** the pretrained I-JEPA ViT-Huge model from `timm` (frozen — no fine-tuning).
2. **Extracts** mean-pooled patch token representations at every transformer block (32 layers) in a single forward pass using PyTorch hooks.
3. **Trains** one `sklearn` Logistic Regression probe per layer on class labels.
4. **Computes** CKA (Centered Kernel Alignment) similarity between all layer pairs.
5. **Visualizes** results:
   - Layer-accuracy curve (val + train)
   - CKA heatmap (layer-to-layer similarity)
   - t-SNE of embeddings at selected layers
   - Summary dashboard

## 🚀 Getting Started

**1. Clone the repository**
```bash
git clone <your-repo-url>
cd Mech-Interp-on-I-JEPA
```

**2. Set up a virtual environment**
```bash
python -m venv venv
.\venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Run the experiment**
```bash
# Default: CIFAR-10, 10 classes, 50 img/class
python run_probing.py

# Override dataset
python run_probing.py --dataset tiny_imagenet --num_classes 20

# Re-use cached features (fast re-run after first extraction)
python run_probing.py --reuse_features

# Skip slow operations
python run_probing.py --no_tsne --no_cka
```

## 📁 Repository Structure

```
├── configs/
│   ├── base_config.yaml          Original causal patching config
│   └── probing_config.yaml       Probing experiment config
├── src/
│   ├── data/
│   │   └── imagenet_loader.py    Flexible dataset loader (CIFAR-10 / Tiny-ImageNet / ImageNet-HF)
│   ├── models/
│   │   └── ijepa_builder.py      I-JEPA ViT-Huge loader via timm
│   ├── experiments/
│   │   ├── patching.py           Causal patching utilities (original)
│   │   └── probing.py            Layer-wise feature extraction + probe training
│   ├── analysis/
│   │   ├── cka.py                Linear CKA similarity metric
│   │   └── visualization.py      Plotting: accuracy curve, CKA heatmap, t-SNE, dashboard
│   └── utils/
│       └── metrics.py            Projection-based feature scoring
├── run_probing.py                Main experiment entrypoint
├── run_causal_trace.py           Original causal patching script
└── requirements.txt
```

## Supported Datasets

| Dataset | Classes | Resolution | Setup |
|---|---|---|---|
| `cifar10` (default) | 10 | 224×224 (upscaled) | Auto-download, ~170 MB |
| `tiny_imagenet` | up to 200 | 224×224 (upscaled) | Auto-download, ~250 MB |
| `imagenet_hf` | up to 1000 | 224×224 | Needs HF token + ~6 GB |

## Output Files

All results are saved to `outputs/probing/`:

| File | Description |
|---|---|
| `probe_results.csv` | Per-layer val/train accuracy + timing |
| `layer_accuracy_curve.png` | Accuracy vs layer depth plot |
| `cka_heatmap.png` | Layer-to-layer CKA similarity heatmap |
| `summary_dashboard.png` | Combined accuracy + CKA figure |
| `tsne_visualization.png` | t-SNE at selected layers |
| `features/` | Cached `.npy` feature arrays (re-use with `--reuse_features`) |

## Expected Results

On CIFAR-10 with I-JEPA ViT-Huge:
- **Early layers (0–5)**: ~15–30% val accuracy (mostly texture / low-level features)  
- **Mid layers (6–20)**: Progressive improvement as semantic features emerge  
- **Late layers (25–31)**: Peak accuracy (typically 70–90%) where high-level semantics dominate  

This matches known SSL behaviour: I-JEPA learns **abstract semantic representations** rather than pixel-level textures.

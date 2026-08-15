# I-JEPA Mechanistic Interpretability Baseline

This repository contains the modularized baseline for causal mediation analysis on the Vision Transformer (ViT-Huge) using the I-JEPA architecture.

For official ArIES club resources, task assignments, and overarching project documentation, please refer to our main Notion workspace.

## 🚀 Getting Started

**1. Clone the repository**
\`\`\`bash
git clone <your-repo-url>
cd aries-mech-interp
\`\`\`

**2. Set up a virtual environment**
We strictly use virtual environments to prevent dependency conflicts.
\`\`\`bash
python -m venv venv
.\venv\Scripts\activate
\`\`\`

**3. Install dependencies**
\`\`\`bash
pip install -r requirements.txt
\`\`\`

## 📁 Repository Structure
* `configs/`: YAML files controlling experiment parameters.
* `src/`: The core engine (models, data pipelines, intervention hooks).
* `notebooks/`: Reserved strictly for exploratory data analysis (EDA) and plotting.
* `data/`: Local storage for generated datasets (ignored by Git).

## 🌿 Branching Strategy
* `main`: Stable, reviewed code only.
* `exp/<name>`: Create a branch using this naming convention for your specific experiments (e.g., `exp/shape-bias`).

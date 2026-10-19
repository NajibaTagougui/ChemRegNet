
This is a live demo at https://NajibaTagougui.github.io/ChemRegApp/

# ChemRegNet

Reference implementation of **ChemRegNet: Transparent Molecular Property Prediction via Cross-Modal Graph–Descriptor Fusion and Scaffold-Aware Contrastive Regularization**.

This repository contains the full training pipeline, the scaffold split construction, the two attribution channels described in the paper, and the scripts that reproduce every table and figure.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

ChemRegNet is a hybrid molecular property prediction model that combines:

1. A three-layer Graph Convolutional Network (GCN) that encodes the molecular graph structure.
2. A three-layer MLP that encodes 200 RDKit physicochemical descriptors.
3. A learnable **cross-modal attention gate** that conditions the descriptor weighting on each molecule's structural context.
4. A **scaffold-aware contrastive regularizer** that shapes the representation space along the axis tested by scaffold splitting.

The model exposes two attribution channels for interpretability:

- **Task saliency** $S_v$: gradient of the scalar prediction with respect to atom features.
- **Activation attribution** $A_v$: gradient of the descriptor-gated pathway with respect to atom features.

The paper reports two contributions: a multi-benchmark measurement of the random-split optimism gap, and a systematic evaluation of ChemRegNet under both scaffold and random splits on five MoleculeNet benchmarks.

---

## Installation

### Requirements

- Python 3.9 or later
- PyTorch 2.2 or later
- PyTorch Geometric 2.5 or later
- RDKit 2024.03 or later
- scikit-learn, scipy, pandas, numpy, matplotlib

No GPU is required. The full pipeline runs on a single 8-core CPU in under 30 minutes per benchmark.

### Setup


# Clone the repository
git clone https://github.com/YOUR_USERNAME/chemregnet.git
cd chemregnet

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt


---

## Quick Start

Train and evaluate on ESOL with the default configuration:


python main.py --dataset esol --seed 42

---

## Datasets

The five benchmarks used in the paper are downloaded at runtime from their public sources. No data files are stored in this repository.

| Dataset | N | Task | Metric | Source |
|---|---|---|---|---|
| ESOL | 1,128 | logS regression | RMSE | DeepChem |
| Lipophilicity | 4,200 | logD regression | RMSE | DeepChem |
| FreeSolv | 642 | hydration free energy | RMSE | MoleculeNet |
| BACE | 1,513 | binary classification | AUROC | MoleculeNet |
| BBBP | 2,039 | binary classification | AUROC | MoleculeNet |

ESOL and Lipophilicity are loaded directly from:

- https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv
- https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv

FreeSolv, BACE, and BBBP are loaded through the MoleculeNet collection.

All five datasets are distributed under open licenses without restrictions.

---

## Project Structure

```
chemregnet/
├── README.md                  <- this file
├── LICENSE                    <- MIT license
├── requirements.txt           <- Python dependencies
├── setup.py                   <- package installation
├── CITATION.cff               <- machine-readable citation
├── .gitignore                 <- files excluded from Git
│
├── main.py                    <- training and evaluation pipeline
├── scaffold_split.py          <- scaffold split construction
├── attribution.py             <- task saliency and activation attribution
├── reproduce_figures.py       <- regenerates all tables and figures
│
├── data/                      <- downloaded data (gitignored)
├── evaluation_outputs/        <- per-seed metrics, attribution values
└── figures/                   <- generated tables and figures
```

---

## Contact

Najiba Tagougui  
Department of Computer Science Engineering, ISIMS  
University of Sfax, Sfax 3025, Tunisia  
Email: najiba.tagougui@isims.usf.tn
```

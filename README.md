# Transferability Predictors

_CS289A Final Project, Spring 2026_

#### Ahmed Mostafa, Junlajak Jongpipattanakul, Malavikha Sudarshan

The goal of this project is to predict, **without any training**, whether a frozen ImageNet-pretrained ResNet-18 will transfer well to a new vision dataset by computing closed-form metrics on cached features. The main result is a side-by-side comparison of three transferability predictors (two label-free, one label-aware) against linear-probe accuracy on six datasets.

This `README.md` file includes instructions on how to reproduce the results from our paper. 

## 1. Project structure

Core files:

| File | Role |
|---|---|
| `data.py` | Builds `DataLoader`s for the six target datasets and the ImageNet reference. Handles resizing to 224×224, ImageNet normalization, and promoting grayscale to 3 channels. |
| `extract.py` | Runs frozen ResNet-18 (ImageNet-1K weights) with a forward hook on `avgpool`, saving 512-d features and labels to `results/features/{dataset}_{split}.npz`. |
| `align.py` | Implements PCA subspace alignment, principal angles, RV coefficient, and H-score. |
| `probe.py` | Standardizes features and fits a multinomial logistic regression (sklearn) as the linear probe; saves probe accuracy and the classifier. |
| `scratch.py` | Trains a randomly initialized ResNet-18 from scratch on each target dataset using SGD + cosine LR; saves scratch accuracy and weights. |
| `plots.py` | Generates all paper figures (scatterplots, principal-angle spectra, variance explained, sensitivity to k, 3×2 metric-comparison grid). |
| `run.py` | Orchestrator: loops over datasets, calls feature extraction, metrics, probe, scratch, writes CSV/JSON, and produces plots. |
| `run_savio.sh` | SLURM submission script for running the heavy parts (especially scratch training) on Berkeley Savio. |

All outputs go under `results/` (features, metrics, models, figures).

## 2. Setup

### 2.1 Environment

```bash
git clone https://github.com/ahmostafa147/Transferability_Predictors.git
cd Transferability_Predictors

python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Requirements:

- Python 3.9+  
- PyTorch + torchvision (with CUDA if you want scratch training)  
- `scikit-learn`, `numpy`, `pandas`  
- `matplotlib` / `seaborn` (for plots)  

To verify GPU:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### 2.2 Datasets

All datasets are automatically downloaded on first use via `torchvision` / `medmnist` and cached locally.

- **CIFAR-10**, **STL-10**, **SVHN**, **EuroSAT** via `torchvision.datasets`  
- **PathMNIST**, **DermaMNIST** via `medmnist`  
- **ImageNet reference**: first 10k images of ImageNet-1K validation, streamed via HuggingFace

By default, torchvision data goes under `./data/` and MedMNIST under `~/.medmnist/`. If you end up changing these locations, make sure to also update the paths in `data.py`.

## 3. How to reproduce the experiments

There are two options:

- **Fast path (recommended):** use cached features and models if present; this reproduces metrics and figures in under a minute once heavy steps have been run once.  
- **Full run from scratch:** recompute features and retrain scratch models, which requires a GPU and takes longer.

### 3.1 End-to-end run (in one command)

From the repo root:

```bash
python run.py
```

What this does:

1. **Reference features**
   - Streams 10k ImageNet validation images through frozen ResNet-18 and stores `imagenet_ref` features.
   - Computes PCA subspaces for k ∈ {10, 20, 50, 100, 200}.

2. **Per target dataset (CIFAR-10, STL-10, SVHN, EuroSAT, PathMNIST, DermaMNIST)**
   1. Extracts train/test features with frozen ResNet-18 → `results/features/{name}_{split}.npz`.
   2. Computes PCA subspace alignment and principal-angle spectra vs. ImageNet reference.
   3. Runs k-sweep alignment for k ∈ {10, 20, 50, 100, 200}.
   4. Computes RV coefficient between target and reference covariance matrices.
   5. Computes H-score using target features and labels.
   6. Fits linear probe (sklearn LogisticRegression) on standardized features → `probe_acc`.
   7. Trains ResNet-18 from scratch for 30 epochs (SGD, momentum 0.9, weight decay 5e-4, cosine schedule, AMP) → `scratch_acc`.
   8. Computes `transfer_gap = probe_acc − scratch_acc`.

3. **Aggregation + plots**
   - Writes `results/metrics.csv` (one row per dataset with all metrics and accuracies).
   - Dumps raw arrays, spectra, and correlations to `results/raw_metrics.json`.
   - Generates all figures into `results/figures/`.

`run.py` is cache-aware: if features/models/accuracies already exist, it reuses them and skips expensive steps.

### 3.2 Running steps manually (if you want finer control)

**(a) Feature extraction**

```bash
python extract.py
```

This will:

- Build loaders for each dataset (train/test + ImageNet ref).  
- Run frozen ResNet-18 with an `avgpool` forward hook.  
- Save `(features, labels)` as compressed NumPy arrays in `results/features/`.

**(b) Metrics (PCA alignment, RV, H-score)**

```bash
python align.py
```

- Computes:
  - PCA subspace alignment at k = 50 (headline) and for k ∈ {10, 20, 50, 100, 200}.  
  - Principal-angle lists per dataset.  
  - RV coefficient per dataset.  
  - H-score per dataset.
- Writes metrics to `results/metrics.csv` and raw arrays to `results/raw_metrics.json`.

**(c) Linear probe**

```bash
python probe.py
```

- Standardizes features, fits `LogisticRegression(max_iter=1000)` on train features.  
- Evaluates on test features → `probe_acc`.  
- Saves classifier under `results/models/probe_{dataset}.pkl` and updates `metrics.csv`.

**(d) Scratch baseline**

```bash
python scratch.py
```

- Trains a randomly initialized ResNet-18 from scratch for 30 epochs on each dataset.  
- Evaluates on test set → `scratch_acc`.  
- Saves weights under `results/models/scratch_{dataset}.pt` and updates `results/scratch_acc.json` and `metrics.csv`.

You can run any subset of these scripts; `run.py` assumes the others are cached if their outputs exist.

## 4. Outputs

Key outputs:

- `results/metrics.csv`  
  - Columns: `dataset, alignment_score, rv_score, h_score, probe_acc, scratch_acc, transfer_gap`  
- `results/raw_metrics.json`  
  - PCA variance curves, principal-angle spectra, alignment k-sweep values, and correlation statistics (Pearson r, Spearman ρ, leave-one-out).  
- `results/features/{name}_{split}.npz`  
  - Cached features and labels for each dataset/split.  
- `results/models/probe_{name}.pkl` / `results/models/scratch_{name}.pt`  
  - Saved probe and scratch models.  

Figures (all under `results/figures/`):

- `metric_comparison.png` – 3×2 grid of each predictor (PCA alignment, RV, H-score) vs. probe accuracy and transfer gap, with correlations annotated.  
- `variance_explained.png` – cumulative PCA variance curves per dataset (intrinsic-dimension confound).  
- `principal_angles.png` – principal-angle spectra vs. ImageNet reference for each dataset.  
- `sensitivity_k.png` – Pearson correlation between PCA alignment and transfer gap as a function of k.  
- `accuracy_bars.png` – probe vs. scratch test accuracy per dataset.  

These reproduce the tables and figures referenced in our report.

## 5. Caching and reruns

To **force recomputation** of a given piece:

- delete `results/features/{name}_{split}.npz` → recomputes features  
- delete `results/models/probe_{name}.pkl` → refits probe  
- delete `results/models/scratch_{name}.pt` or remove entry from `results/scratch_acc.json` → retrains scratch model  
- delete `results/metrics.csv` / `results/raw_metrics.json` → recomputes metrics and correlations  
- delete files in `results/figures/` → regenerates plots on next `run.py` / `plots.py`

`run.py` only skips a scratch run if **both** the saved weights and the cached accuracy entry exist, so you never end up with a missing model for a supposedly cached run.

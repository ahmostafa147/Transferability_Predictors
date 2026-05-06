import os
import csv
import json
import random
import numpy as np
import torch

from data import get_loader, num_classes
from extract import cached_features, imagenet_probs
from align import pca_subspace, alignment_score, principal_angles, rv_coefficient, h_score, leep_score
from probe import linear_probe
from scratch import train_from_scratch
from plots import (scatter_metric_vs_gap, principal_angle_spectra,
                   accuracy_bars, variance_explained, sensitivity_to_k,
                   metric_comparison_grid)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DATASETS = ["cifar10", "stl10", "svhn", "eurosat", "pathmnist", "dermamnist"]
K_MAIN = 50
K_SWEEP = [10, 20, 50, 100, 200]
SCRATCH_CACHE = "results/scratch_acc.json"
os.makedirs("results", exist_ok=True)


def load_scratch_cache():
    if os.path.exists(SCRATCH_CACHE):
        with open(SCRATCH_CACHE) as f:
            return json.load(f)
    return {}


def save_scratch_cache(cache):
    with open(SCRATCH_CACHE, "w") as f:
        json.dump(cache, f, indent=2)


def main():
    print("[ref] extracting imagenet_ref features")
    F_ref, _ = cached_features("imagenet_ref", "train")
    print(f"[ref] done, shape={F_ref.shape}; computing reference PCA subspaces")
    U_ref_by_k = {k: pca_subspace(F_ref, k) for k in K_SWEEP}

    rows = []
    angle_spectra = {}
    feat_for_var = {"imagenet_ref": F_ref}
    scores_by_k = {k: [] for k in K_SWEEP}
    scratch_cache = load_scratch_cache()

    for i, name in enumerate(DATASETS, 1):
        print(f"\n=== [{i}/{len(DATASETS)}] {name} ===")
        print(f"[{name}] extracting train features")
        F_tr, y_tr = cached_features(name, "train")
        print(f"[{name}] extracting test features")
        F_te, y_te = cached_features(name, "test")
        feat_for_var[name] = F_tr

        print(f"[{name}] computing PCA + alignment (k={K_MAIN})")
        U = pca_subspace(F_tr, K_MAIN)
        score = alignment_score(U, U_ref_by_k[K_MAIN])
        angle_spectra[name] = principal_angles(U, U_ref_by_k[K_MAIN])

        print(f"[{name}] computing RV coefficient + H-score")
        rv = rv_coefficient(F_tr, F_ref)
        h = h_score(F_tr, y_tr)

        print(f"[{name}] computing LEEP")
        theta = imagenet_probs(F_tr)
        leep = leep_score(theta, y_tr)

        print(f"[{name}] sensitivity sweep over k={K_SWEEP}")
        for k in K_SWEEP:
            Uk = pca_subspace(F_tr, k)
            scores_by_k[k].append(alignment_score(Uk, U_ref_by_k[k]))

        print(f"[{name}] training linear probe")
        probe_acc = linear_probe(F_tr, y_tr, F_te, y_te,
                                 save_path=f"results/models/probe_{name}.pkl")

        scratch_path = f"results/models/scratch_{name}.pt"
        if name in scratch_cache and os.path.exists(scratch_path):
            scratch_acc = scratch_cache[name]
            print(f"[{name}] cached scratch_acc={scratch_acc:.4f} (model at {scratch_path})")
        else:
            print(f"[{name}] training from scratch")
            train_loader = get_loader(name, "train", batch_size=256, augment=True)
            test_loader = get_loader(name, "test", batch_size=512, shuffle=False)
            scratch_acc = train_from_scratch(train_loader, test_loader, num_classes(name),
                                             save_path=scratch_path)
            scratch_cache[name] = scratch_acc
            save_scratch_cache(scratch_cache)

        rows.append({
            "dataset": name,
            "alignment_score": score,
            "rv_score": rv,
            "h_score": h,
            "leep_score": leep,
            "probe_acc": probe_acc,
            "scratch_acc": scratch_acc,
            "transfer_gap": probe_acc - scratch_acc,
        })
        print(f"[{name}] result: {rows[-1]}")

    print("\n[save] writing results/metrics.csv")
    with open("results/metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    names = [r["dataset"] for r in rows]
    scores = [r["alignment_score"] for r in rows]
    probe_accs = [r["probe_acc"] for r in rows]
    scratch_accs = [r["scratch_acc"] for r in rows]
    gaps = [r["transfer_gap"] for r in rows]

    print("[plots] generating figures")
    scatter_metric_vs_gap(scores, gaps, names)
    principal_angle_spectra(angle_spectra)
    accuracy_bars(probe_accs, scratch_accs, names)
    variance_explained(feat_for_var)
    sensitivity_to_k(scores_by_k, gaps, K_SWEEP)
    correlations = metric_comparison_grid(rows, leave_out="stl10")

    print("[save] writing results/raw_metrics.json")
    raw = {
        "config": {"seed": SEED, "k_main": K_MAIN, "k_sweep": K_SWEEP, "datasets": DATASETS},
        "angle_spectra": {k: v.tolist() for k, v in angle_spectra.items()},
        "scores_by_k": {str(k): scores_by_k[k] for k in K_SWEEP},
        "correlations": correlations,
    }
    with open("results/raw_metrics.json", "w") as f:
        json.dump(raw, f, indent=2)
    print("[done]")


if __name__ == "__main__":
    main()

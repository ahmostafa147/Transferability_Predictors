import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

os.makedirs("results/figures", exist_ok=True)


def scatter_alignment_vs_gap(scores, gaps, names, path="results/figures/alignment_vs_gap.png"):
    scores = np.asarray(scores)
    gaps = np.asarray(gaps)
    r, p = pearsonr(scores, gaps)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(scores, gaps, s=60)
    for x, y, n in zip(scores, gaps, names):
        ax.annotate(n, (x, y), xytext=(5, 5), textcoords="offset points", fontsize=9)
    m, b = np.polyfit(scores, gaps, 1)
    xs = np.linspace(scores.min(), scores.max(), 100)
    ax.plot(xs, m * xs + b, "r--", alpha=0.7)
    ax.set_xlabel("PCA alignment score")
    ax.set_ylabel("Transfer gap (probe - scratch)")
    ax.set_title(f"Alignment vs transfer gap (r={r:.2f}, p={p:.3f})")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def principal_angle_spectra(angles_dict, path="results/figures/principal_angles.png"):
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, ang in angles_dict.items():
        ax.plot(ang, label=name)
    ax.set_xlabel("Index")
    ax.set_ylabel("Principal angle (deg)")
    ax.set_title("Principal angle spectra vs ImageNet reference")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def accuracy_bars(probe_accs, scratch_accs, names, path="results/figures/accuracy_bars.png"):
    x = np.arange(len(names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w / 2, probe_accs, w, label="Linear probe")
    ax.bar(x + w / 2, scratch_accs, w, label="From scratch")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Test accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def variance_explained(features_dict, path="results/figures/variance_explained.png"):
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, F in features_dict.items():
        Fc = F - F.mean(0)
        s = np.linalg.svd(Fc, compute_uv=False)
        ev = (s ** 2)
        cum = np.cumsum(ev) / ev.sum()
        ax.plot(cum, label=name)
    ax.set_xlabel("Component")
    ax.set_ylabel("Cumulative variance explained")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def sensitivity_to_k(scores_by_k, gaps, k_values, path="results/figures/sensitivity_k.png"):
    rs = []
    for k in k_values:
        r, _ = pearsonr(scores_by_k[k], gaps)
        rs.append(r)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(k_values, rs, "o-")
    ax.set_xlabel("k (PCA components)")
    ax.set_ylabel("Pearson r (alignment, gap)")
    ax.set_title("Sensitivity of correlation to k")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def metric_comparison_grid(rows, path="results/figures/metric_comparison.png", leave_out=None):
    metrics = [("alignment_score", "PCA alignment"),
               ("rv_score", "RV coefficient"),
               ("h_score", "H-score")]
    targets = [("probe_acc", "Probe accuracy"),
               ("transfer_gap", "Transfer gap")]
    names = [r["dataset"] for r in rows]
    fig, axes = plt.subplots(len(targets), len(metrics), figsize=(15, 9))
    summary = {}
    for i, (tkey, tlabel) in enumerate(targets):
        y = np.array([r[tkey] for r in rows])
        for j, (mkey, mlabel) in enumerate(metrics):
            x = np.array([r[mkey] for r in rows])
            ax = axes[i, j]
            ax.scatter(x, y, s=50)
            for xi, yi, n in zip(x, y, names):
                ax.annotate(n, (xi, yi), xytext=(4, 4), textcoords="offset points", fontsize=7)
            m, b = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 50)
            ax.plot(xs, m * xs + b, "r--", alpha=0.5)
            r_p, p_p = pearsonr(x, y)
            r_s, p_s = spearmanr(x, y)
            title = f"{mlabel} vs {tlabel}\nPearson={r_p:.2f} (p={p_p:.2f})  Spearman={r_s:.2f}"
            entry = {"pearson_r": float(r_p), "pearson_p": float(p_p),
                     "spearman_r": float(r_s), "spearman_p": float(p_s)}
            if leave_out and leave_out in names:
                mask = np.array([n != leave_out for n in names])
                if mask.sum() >= 3:
                    r_loo, p_loo = pearsonr(x[mask], y[mask])
                    s_loo, _ = spearmanr(x[mask], y[mask])
                    title += f"\nLOO({leave_out}): Pearson={r_loo:.2f} Spearman={s_loo:.2f}"
                    entry["pearson_r_loo"] = float(r_loo)
                    entry["spearman_r_loo"] = float(s_loo)
            ax.set_title(title, fontsize=9)
            ax.set_xlabel(mlabel)
            ax.set_ylabel(tlabel)
            summary[f"{mkey}_vs_{tkey}"] = entry
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return summary

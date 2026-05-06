import numpy as np


def pca_subspace(F, k):
    Fc = F - F.mean(axis=0, keepdims=True)
    cov = (Fc.T @ Fc) / (Fc.shape[0] - 1)
    w, V = np.linalg.eigh(cov)
    idx = np.argsort(w)[::-1][:k]
    return V[:, idx]


def alignment_score(U1, U2):
    s = np.linalg.svd(U1.T @ U2, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    return float(s.mean())


def principal_angles(U1, U2):
    s = np.linalg.svd(U1.T @ U2, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    return np.degrees(np.arccos(s))


def rv_coefficient(F1, F2):
    F1c = F1 - F1.mean(0, keepdims=True)
    F2c = F2 - F2.mean(0, keepdims=True)
    C1 = F1c.T @ F1c
    C2 = F2c.T @ F2c
    num = float(np.sum(C1 * C2))
    den = float(np.sqrt(np.sum(C1 * C1) * np.sum(C2 * C2)))
    return num / den


def h_score(F, y):
    F = np.asarray(F, dtype=np.float64)
    y = np.asarray(y).flatten()
    mu = F.mean(0)
    Fc = F - mu
    cov_f = (Fc.T @ Fc) / (F.shape[0] - 1)
    classes = np.unique(y)
    means = np.stack([F[y == c].mean(0) for c in classes])
    n_per = np.array([(y == c).sum() for c in classes], dtype=np.float64)
    p = n_per / n_per.sum()
    Mc = means - mu
    cov_g = (Mc * p[:, None]).T @ Mc
    return float(np.trace(np.linalg.pinv(cov_f) @ cov_g))


# LEEP (Nguyen et al., ICML 2020). LLM-assisted for writing vectorization.
def leep_score(theta, y):
    n, Z = theta.shape
    y = np.asarray(y).flatten()
    Y = int(y.max()) + 1
    joint = np.zeros((Y, Z), dtype=np.float64)
    np.add.at(joint, y, theta)
    joint /= n
    p_z = joint.sum(0, keepdims=True)
    cond = joint / np.maximum(p_z, 1e-12)
    p_yi = (theta * cond[y]).sum(1)
    return float(np.log(np.maximum(p_yi, 1e-12)).mean())

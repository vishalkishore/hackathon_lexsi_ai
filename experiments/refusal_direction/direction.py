"""Difference-in-means refusal direction, separation statistics and the ablation / addition edits (pure torch)."""
import random

import torch


def fit_direction(harmful, benign):
    """harmful, benign: [n, ..., d] activations. Returns (unit direction, length of the mean difference), each [..., d] / [...]."""
    r = harmful.float().mean(0) - benign.float().mean(0)
    norm = r.norm(dim=-1)
    return r / norm.unsqueeze(-1), norm


def cohen_d(a, b):
    """Standardized mean difference (pooled SD) along the first axis."""
    a, b = a.float(), b.float()
    na, nb = a.shape[0], b.shape[0]
    pooled = (((na - 1) * a.var(0) + (nb - 1) * b.var(0)) / (na + nb - 2)).sqrt()
    return (a.mean(0) - b.mean(0)) / pooled


def separation(p_harmful, p_benign):
    """Projections [n, ...] -> raw gap and Cohen's d per remaining axis."""
    return {'gap': p_harmful.float().mean(0) - p_benign.float().mean(0), 'd': cohen_d(p_harmful, p_benign)}


def bootstrap_ratio(p, q, n_boot=2000, seed=0):
    """95% interval for gap(p)/gap(q); p, q: {group_id: (harmful projection, benign projection)} on shared groups."""
    keys = sorted(set(p) & set(q))
    rng = random.Random(seed)

    def ratio(ks):
        gp = sum(p[k][0] - p[k][1] for k in ks)
        gq = sum(q[k][0] - q[k][1] for k in ks)
        return gp / gq

    boots = sorted(ratio([rng.choice(keys) for _ in keys]) for _ in range(n_boot))
    return ratio(keys), [boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]]


def project_out(h, r_hat):
    """Remove the r_hat component from every vector: h - (h . r_hat) r_hat."""
    r = r_hat.to(h.dtype)
    return h - (h @ r).unsqueeze(-1) * r

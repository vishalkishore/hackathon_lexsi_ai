"""Statistics: one paired bootstrap (by group_id, 4000, seed 0), exact McNemar, Holm; Wilson interval for stored rates."""
import math

import numpy as np


def boot(a: dict, b: dict, n: int = 4000, seed: int = 0):
    """Paired bootstrap of mean(a - b) over shared group ids. a, b: {group_id: 0/1}. Returns points
    (delta, ci_low, ci_high, n_groups) in percentage points, or None if no shared group."""
    g = sorted(a.keys() & b.keys())
    if not g:
        return None
    d = np.array([a[x] - b[x] for x in g], float)
    bs = d[np.random.default_rng(seed).integers(0, len(d), (n, len(d)))].mean(1)
    return 100 * d.mean(), 100 * np.quantile(bs, 0.025), 100 * np.quantile(bs, 0.975), len(g)


def mcnemar(a: dict, b: dict) -> float:
    """Exact two-sided McNemar p on paired 0/1 outcomes (shared group ids)."""
    g = a.keys() & b.keys()
    n10 = sum(a[x] == 1 and b[x] == 0 for x in g)
    n01 = sum(a[x] == 0 and b[x] == 1 for x in g)
    n = n10 + n01
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(n10, n01) + 1)) / 2 ** n)


def holm(ps: list) -> list:
    """Holm step-down adjusted p-values, input order. Family = the list given (one table)."""
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


def wilson(rate_pct: float, n: int, z: float = 1.959964) -> tuple:
    """Wilson 95% interval (percent) for a stored rate over n prompts."""
    p = rate_pct / 100
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return 100 * (c - h), 100 * (c + h)

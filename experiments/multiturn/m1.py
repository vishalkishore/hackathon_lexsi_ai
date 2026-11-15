"""Stage M1: single-pass context-invariant readout at layer 19 (score level, CPU, fp64). Uses the M0 cache only.

Directions (unit norm, harmful side positive):
  (a) ordinary: w = (S_W    + lam S_C + eps I)^-1 (mu_H - mu_B)
  (b) paired:   w = (S_pair + lam S_C + eps I)^-1 dbar,  d_i = h(p,1,s,d) - h(p,0,s,d)
  S_C = (1/N) sum Delta Delta^T, Delta = h(p,a,s,d) - h(p,a,s,0), d in {2, 5, 8}, uncentered.
  Each matrix is divided by tr/4096 (mean eigenvalue 1) before mixing. eps from Ledoit-Wolf (closed form): with
  shrinkage delta, eps = delta / (1 - delta), so lam = 0 gives the Ledoit-Wolf estimate up to a scale.
  Speed: fold statistics are all-pairs sums minus test-fold sums; per (fold, version, S_C) one Cholesky and one
  eigendecomposition give w for every lam (LambdaPath). A direct fp64 solve checks the eigen path on fold 0.
  Control: S_C from shuffled depth pairs, Delta = h(p,a,s,d) - h(pi(p),a,s,0).
Scores: s_ctx = r_hat . h, s_inv = w . h, s_iso = r_hat . h of the same request at depth 0 (single-turn reference).
5-fold CV by family (= pair). Catch = TPR at the benign 95th percentile of the same (script, depth) cell.
GO rule: NOTES.md, section M1 (written before this stage ran).
"""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "refusal_direction"))

from cache import DEPTHS, LAYERS, SCRIPTS, environment  # noqa: E402
from ov_common import SCORE_LAYER, auroc, catch_rate, folds, r_hat  # noqa: E402
from prompts import ROOT  # noqa: E402

LAMBDAS = (0.0, 0.1, 1.0, 10.0, 100.0)
VERSIONS = ("a", "b")
MODELS = ("base", "read")
EN, DEVA, LATN, ENHI = (SCRIPTS.index(s) for s in ("en", "hi-Deva", "hi-Latn", "en-hi"))
D8 = DEPTHS.index(8)
T4 = 2.7764451051977987  # t quantile 0.975, 4 degrees of freedom (5 folds)


def scaled(S):
    return S / (torch.trace(S) / S.shape[0])


def lw_shrinkage(scatter, n, quartic):
    """Ledoit-Wolf shrinkage from its closed form (same formula as sklearn.covariance.ledoit_wolf_shrinkage with
    assume_centered=True), given the centred scatter matrix sum (x - m)(x - m)^T over n rows and
    quartic = sum ||x - m||^4. The shrinkage is scale-free, so it is computed before the trace scaling."""
    p = scatter.shape[0]
    emp = scatter / n
    trace = float(torch.trace(emp))
    mu = trace / p
    delta_ = float((emp ** 2).sum())
    beta = (quartic / n - delta_) / (p * n)
    delta = (delta_ - 2 * mu * trace + p * mu ** 2) / p
    beta = min(beta, delta)
    shrink = 0.0 if beta == 0 else beta / delta
    if shrink >= 1:
        raise ValueError("Ledoit-Wolf shrinkage is 1: the covariance carries no structure, eps is undefined")
    return shrink


def eps_of(shrink):
    """With mean eigenvalue 1 after scaling, (1 - delta) S + delta I is proportional to S + eps I, eps = delta / (1 - delta)."""
    return shrink / (1 - shrink)


def pair_rows(X, idx, perm=None):
    """Row sets of the pairs idx. X: [P, A, S, D, F]. perm: shuffled depth-0 partner for the control S_C."""
    F = X.shape[-1]
    Xi = X[idx]
    base = Xi[:, :, :, :1] if perm is None else X[perm[idx]][:, :, :, :1]
    return {"harm": Xi[:, 1].reshape(-1, F), "ben": Xi[:, 0].reshape(-1, F),
            "d": (Xi[:, 1] - Xi[:, 0]).reshape(-1, F), "delta": (Xi[:, :, :, 1:] - base).reshape(-1, F)}


def sums(rows):
    return {k: (R.T @ R, R.sum(0), R.shape[0]) for k, R in rows.items()}


def minus(total, part):
    return {k: (total[k][0] - part[k][0], total[k][1] - part[k][1], total[k][2] - part[k][2]) for k in total}


def centred(G, s, n):
    m = s / n
    return G - n * torch.outer(m, m), m


def quartic(R, m):
    return float((((R - m) ** 2).sum(1) ** 2).sum())


def fold_matrices(st, rows):
    """Scaled matrices from (train) sums st; rows = the same train rows (for the Ledoit-Wolf quartic term)."""
    scat_h, mu_h = centred(*st["harm"])
    scat_b, mu_b = centred(*st["ben"])
    n_w = st["harm"][2] + st["ben"][2]
    scat_w = scat_h + scat_b
    q_w = quartic(rows["harm"], mu_h) + quartic(rows["ben"], mu_b)
    shrink_a = lw_shrinkage(scat_w, n_w, q_w)

    scat_d, dbar = centred(*st["d"])
    shrink_b = lw_shrinkage(scat_d, st["d"][2], quartic(rows["d"], dbar))

    S_C = st["delta"][0] / st["delta"][2]
    return {"a": (scaled(scat_w / (n_w - 2)), mu_h - mu_b, eps_of(shrink_a), shrink_a),
            "b": (scaled(scat_d / (st["d"][2] - 1)), dbar, eps_of(shrink_b), shrink_b),
            "S_C": scaled(S_C)}


def matrices(X, train, perm=None):
    """Direct (reference) construction from the train rows. Used for the final refit, M1c and the tests."""
    rows = pair_rows(X, train, perm)
    mats = fold_matrices(sums(rows), rows)
    mats["train_rows"] = (rows["harm"], rows["ben"])
    return mats


def solve(mats, version, lam):
    """Direct fp64 solve of (S + lam S_C + eps I) w = b."""
    S, b, eps, _ = mats[version]
    M = S + lam * mats["S_C"] + eps * torch.eye(S.shape[0], dtype=S.dtype)
    w_raw = torch.linalg.solve(M, b)
    resid = float((M @ w_raw - b).norm() / b.norm())
    return w_raw / w_raw.norm(), M, b, resid


class LambdaPath:
    """All lam at once. S + eps I = L L^T; C~ = L^-1 S_C L^-T = V diag(Lam) V^T, so
    S + eps I + lam S_C = L V (I + lam Lam) V^T L^T and w(lam) = L^-T V (I + lam Lam)^-1 V^T L^-1 b."""

    def __init__(self, S, eps, b, S_C):
        A = S + eps * torch.eye(S.shape[0], dtype=S.dtype)
        self.L = torch.linalg.cholesky(A)
        Y = torch.linalg.solve_triangular(self.L, S_C, upper=False)          # L^-1 S_C
        C = torch.linalg.solve_triangular(self.L, Y.T, upper=False)          # L^-1 S_C L^-T
        self.lam_eig, self.V = torch.linalg.eigh((C + C.T) / 2)
        self.c = self.V.T @ torch.linalg.solve_triangular(self.L, b[:, None], upper=False)[:, 0]

    def w(self, lam):
        z = self.V @ (self.c / (1 + lam * self.lam_eig))
        w_raw = torch.linalg.solve_triangular(self.L.T, z[:, None], upper=True)[:, 0]
        return w_raw


def rayleigh(w, M, b):
    return float((w @ b) ** 2 / (w @ M @ w))


def evaluate(oof, fold_index):
    """oof: dict name -> scores [P, A, S, D]. Pooled OOF AUROC/catch per cell, and per-fold catch at d = 8."""
    out = {}
    for name, sc in oof.items():
        cells = {}
        for s in range(4):
            for d in range(4):
                a, c = auroc(sc[:, 1, s, d].tolist(), sc[:, 0, s, d].tolist()), \
                    catch_rate(sc[:, 1, s, d].tolist(), sc[:, 0, s, d].tolist())
                per_fold = [catch_rate(sc[fold_index == f, 1, s, d].tolist(), sc[fold_index == f, 0, s, d].tolist())
                            for f in range(5)]
                cells[f"{SCRIPTS[s]}|{DEPTHS[d]}"] = {"auroc": a, "catch": c, "catch_per_fold": per_fold}
        out[name] = cells
    return out


def recovery(m, script):
    key = f"{script}|8"
    ctx, inv, iso = (m[n][key]["catch"] for n in ("ctx", "inv", "iso"))
    R = (inv - ctx) / (iso - ctx) if iso != ctx else float("nan")
    per_fold = []
    for f in range(5):
        c, i, s = (m[n][key]["catch_per_fold"][f] for n in ("ctx", "inv", "iso"))
        per_fold.append((i - c) / (s - c) if s != c else float("nan"))
    finite = [x for x in per_fold if math.isfinite(x)]
    if len(finite) >= 2:
        mean, sd = float(np.mean(finite)), float(np.std(finite, ddof=1))
        half = T4 * sd / math.sqrt(len(finite))
        ci = (mean - half, mean + half)
    else:
        ci = (float("nan"), float("nan"))
    return {"R": R, "R_per_fold": per_fold, "fold_ci": ci}


def run_model(X, pairs, r):
    by_id = folds(pairs)
    fold_index = np.array([by_id[p] for p in pairs])
    gen = torch.Generator().manual_seed(0)
    perm = torch.randperm(len(pairs), generator=gen).numpy()
    F = X.shape[-1]

    s_ctx = torch.einsum("pasdf,f->pasd", X, r)
    s_iso = s_ctx[:, :, :, :1].expand_as(s_ctx).clone()
    oof = {(v, lam, ctrl): torch.zeros_like(s_ctx) for v in VERSIONS for lam in LAMBDAS for ctrl in (False, True)}
    checks = {"M1a_residual": [], "M1b_rayleigh": [], "M1c_lda": None, "eigen_vs_direct": []}
    info = []

    # Fold statistics: all-pairs sums minus the test-fold sums (one pass over the data).
    every = np.arange(len(pairs))
    test_sums = {f: sums(pair_rows(X, np.where(fold_index == f)[0])) for f in range(5)}
    test_sums_ctrl = {f: sums({"delta": pair_rows(X, np.where(fold_index == f)[0], perm)["delta"]}) for f in range(5)}
    total = {k: tuple(sum(test_sums[f][k][i] for f in range(5)) for i in range(3)) for k in test_sums[0]}
    total_ctrl = {"delta": tuple(sum(test_sums_ctrl[f]["delta"][i] for f in range(5)) for i in range(3))}
    assert total["harm"][2] == len(every) * 16

    for f in range(5):
        train = np.where(fold_index != f)[0]
        test = np.where(fold_index == f)[0]
        rows = pair_rows(X, train)
        st = minus(total, test_sums[f])
        mats = fold_matrices(st, rows)
        S_C_ctrl = scaled(minus(total_ctrl, test_sums_ctrl[f])["delta"][0] / st["delta"][2])
        info.append({"fold": f, "shrink_a": mats["a"][3], "shrink_b": mats["b"][3],
                     "eps_a": mats["a"][2], "eps_b": mats["b"][2]})
        gen_r = torch.Generator().manual_seed(f)
        rand = torch.randn(F, 100, dtype=X.dtype, generator=gen_r)
        rand = rand / rand.norm(dim=0, keepdim=True)

        for v in VERSIONS:
            S, b, eps, _ = mats[v]
            for ctrl in (False, True):
                S_C = S_C_ctrl if ctrl else mats["S_C"]
                path = LambdaPath(S, eps, b, S_C)
                W = []
                for lam in LAMBDAS:
                    w_raw = path.w(lam)
                    M_w = S @ w_raw + eps * w_raw + lam * (S_C @ w_raw)
                    checks["M1a_residual"].append(float((M_w - b).norm() / b.norm()))
                    w = w_raw / w_raw.norm()
                    if not ctrl:
                        # M1b: w maximises (u.b)^2 / u^T M u; compare with 100 random unit vectors and r_hat.
                        U = torch.cat([rand, r[:, None]], 1)
                        MU = S @ U + eps * U + lam * (S_C @ U)
                        J_other = float(((U.T @ b) ** 2 / (U * MU).sum(0)).max())
                        J = float((w @ b) ** 2 / (w @ (S @ w + eps * w + lam * (S_C @ w))))
                        checks["M1b_rayleigh"].append(J >= J_other)
                    if f == 0 and lam == 10.0:
                        direct = solve({v: mats[v], "S_C": S_C}, v, lam)[0]
                        checks["eigen_vs_direct"].append(float(w @ direct))
                    W.append(w)
                proj = torch.einsum("pasdf,fl->pasdl", X[test], torch.stack(W, 1))
                for li, lam in enumerate(LAMBDAS):
                    oof[(v, lam, ctrl)][test] = proj[..., li]
        if f == 0:
            mats["train_rows"] = (rows["harm"], rows["ben"])
            checks["M1c_lda"] = lda_check(mats)
        print(f"fold {f} done", flush=True)

    results = {}
    for key, sc in oof.items():
        m = evaluate({"ctx": s_ctx, "inv": sc, "iso": s_iso}, fold_index)
        results[key] = {"metrics": m, "R": {s: recovery(m, s) for s in ("hi-Latn", "en-hi")}}
    return results, checks, info, oof, s_ctx, s_iso, fold_index


def lda_check(mats):
    """lam = 0, eps -> 0 limit of (a) against sklearn LDA (least squares solver, no shrinkage)."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    harm, ben = mats["train_rows"]
    S, b, _, _ = mats["a"]
    w = torch.linalg.solve(S, b)
    w = w / w.norm()
    lda = LinearDiscriminantAnalysis(solver="lsqr").fit(torch.cat([ben, harm]).numpy(),
                                                        np.r_[np.zeros(len(ben)), np.ones(len(harm))])
    coef = torch.tensor(lda.coef_[0], dtype=w.dtype)
    cos = float(w @ coef / coef.norm())
    return {"cos": cos, "tol": 0.999, "pass": cos >= 0.999}


def bootstrap_R(s_ctx, s_inv, s_iso, script, n_boot=2000, seed=0):
    """Pair bootstrap of R at depth 8: resample pairs with replacement; in each resample the benign 95th percentile
    and the catch rates are recomputed for ctx, inv and iso. Resamples with iso catch = ctx catch are dropped (R
    undefined) and counted. Returns the percentile 95% CI."""
    s, d = SCRIPTS.index(script), DEPTHS.index(8)
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, s_ctx.shape[0], (n_boot, s_ctx.shape[0]), generator=gen)

    def catch(sc):
        harm, ben = sc[:, 1, s, d][idx], sc[:, 0, s, d][idx]
        t = torch.quantile(ben, 0.95, dim=1, keepdim=True)
        return (harm > t).double().mean(1)

    c_ctx, c_inv, c_iso = catch(s_ctx), catch(s_inv), catch(s_iso)
    ok = c_iso != c_ctx
    R = (c_inv[ok] - c_ctx[ok]) / (c_iso[ok] - c_ctx[ok])
    lo, hi = torch.quantile(R, torch.tensor([0.025, 0.975], dtype=R.dtype)).tolist()
    return {"ci": (lo, hi), "n_boot": n_boot, "n_undefined": int((~ok).sum()), "seed": seed}


def go_rule(results, oof, s_ctx, s_iso):
    """GO (NOTES.md M1, amended 2026-11-11 before M1 ran). Best (version, lam) by mean pooled OOF catch_inv at
    d = 8 over hi-Latn and en-hi (unchanged)."""
    def score(key):
        m = results[key]["metrics"]["inv"]
        return (m["hi-Latn|8"]["catch"] + m["en-hi|8"]["catch"]) / 2

    best = max(((v, lam, False) for v in VERSIONS for lam in LAMBDAS), key=score)
    res = results[best]
    R = res["R"]["hi-Latn"]
    boot = bootstrap_R(s_ctx, oof[best], s_iso, "hi-Latn")
    m = res["metrics"]
    drops = {f"{s}|{d}": m["ctx"][f"{s}|{d}"]["auroc"] - m["inv"][f"{s}|{d}"]["auroc"]
             for s in ("en", "hi-Deva") for d in DEPTHS}
    mean_drop = {s: float(np.mean([drops[f"{s}|{d}"] for d in DEPTHS])) for s in ("en", "hi-Deva")}
    ctrl_R = results[(best[0], best[1], True)]["R"]["hi-Latn"]["R"]
    rules = {
        "R_hi-Latn >= 0.5": R["R"] >= 0.5,
        "pair-bootstrap CI (2000) lower bound > 0.1": boot["ci"][0] > 0.1,
        "en and hi-Deva mean AUROC drop over depths <= 0.01": all(x <= 0.01 for x in mean_drop.values()),
        "en and hi-Deva AUROC drop <= 0.02 in every cell": all(x <= 0.02 for x in drops.values()),
        "shuffled-control R < 0.2": ctrl_R < 0.2,
    }
    original = {  # the rule before the amendment, reported for comparison only
        "R_hi-Latn >= 0.5": R["R"] >= 0.5,
        "fold-CI lower bound > 0.3": R["fold_ci"][0] > 0.3,
        "en/hi-Deva AUROC drop <= 0.01 at every d": all(x <= 0.01 for x in drops.values()),
        "shuffled-control R < 0.2": ctrl_R < 0.2,
    }
    return {"best": {"version": best[0], "lambda": best[1]}, "selection_score": score(best),
            "R_hi-Latn": R, "R_hi-Latn_bootstrap": boot, "R_en-hi": res["R"]["en-hi"],
            "auroc_drop_en_hiDeva": drops, "mean_auroc_drop_en_hiDeva": mean_drop,
            "control_R_hi-Latn": ctrl_R, "rules": rules, "GO": all(rules.values()),
            "rule_version": "amended 2026-11-11 (NOTES.md), fixed before M1 ran",
            "original_rule_for_comparison": {"rules": original, "GO": all(original.values())}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=48)
    parser.add_argument("--notes", type=Path, default=ROOT / "NOTES.md", help="where a VERIFY failure is recorded")
    args = parser.parse_args()

    from setproctitle import setproctitle
    setproctitle("aya_alignment:M1")
    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "env-m1.json").write_text(json.dumps(environment(args.out) | {"cache": str(args.cache)}, indent=1))

    pairs = json.loads((args.cache / "pairs.json").read_text())
    r = r_hat("cpu").double()
    li = LAYERS.index(SCORE_LAYER)
    summary, frames, all_checks = {}, [], {}

    for name in MODELS:
        X = torch.load(args.cache / name / "h.pt")[..., li, :].double()
        results, checks, info, oof, s_ctx, s_iso, fold_index = run_model(X, pairs, r)
        go = go_rule(results, oof, s_ctx, s_iso)
        resid = max(checks["M1a_residual"])
        all_checks[name] = {
            "M1a_residual": {"max": resid, "tol": 1e-10, "pass": resid < 1e-10},
            "M1b_rayleigh": {"n": len(checks["M1b_rayleigh"]), "pass": all(checks["M1b_rayleigh"])},
            "M1c_lda": checks["M1c_lda"],
            "eigen_vs_direct": {"min_cos": min(checks["eigen_vs_direct"]), "tol": 1 - 1e-10,
                                "pass": min(checks["eigen_vs_direct"]) >= 1 - 1e-10},
        }
        summary[name] = {"go": go, "fits": info,
                         "table": {f"{v}|{lam:g}|{'control' if c else 'real'}": res for (v, lam, c), res in results.items()}}

        for (v, lam, ctrl), sc in oof.items():
            for p in range(len(pairs)):
                for a in range(2):
                    for s in range(4):
                        for d in range(4):
                            frames.append((name, v, lam, ctrl, pairs[p], a, SCRIPTS[s], DEPTHS[d], int(fold_index[p]),
                                           float(s_ctx[p, a, s, d]), float(sc[p, a, s, d]), float(s_iso[p, a, s, d])))

        if name == "read":
            # Refit the selected configuration on all pairs.
            mats = matrices(X, np.arange(len(pairs)))
            w, _, _, resid_all = solve(mats, go["best"]["version"], go["best"]["lambda"])
            torch.save({"w": w, "version": go["best"]["version"], "lambda": go["best"]["lambda"],
                        "eps": mats[go["best"]["version"]][2], "layer": SCORE_LAYER, "n_pairs": len(pairs),
                        "model": "read (delta_final, gamma 1)", "residual": resid_all,
                        "note": "s_inv = w . hidden_states[19] at the last prompt token"}, args.out / "w_final.pt")

    pd.DataFrame(frames, columns=["model", "version", "lambda", "control", "pair_id", "harmful", "script", "depth",
                                  "fold", "s_ctx", "s_inv", "s_iso"]).to_parquet(args.out / "oof_scores.parquet")
    (args.out / "results.json").write_text(json.dumps(summary, default=str))
    (args.out / "checks.json").write_text(json.dumps(all_checks, indent=1))
    (args.out / "go.json").write_text(json.dumps(summary["read"]["go"], indent=1, default=str))
    write_summary(args.out, summary, all_checks)

    failed = [f"{n}:{k}" for n, c in all_checks.items() for k, v in c.items() if not v["pass"]]
    if failed:
        with args.notes.open("a") as f:
            f.write(f"\n## M1 VERIFY FAILED ({datetime.now(timezone.utc).isoformat()})\n"
                    f"Output {args.out}. Failed: {', '.join(failed)}. Stage stopped.\n")
        sys.exit(f"VERIFY failed: {failed}")
    print("M1 GO (read model):", summary["read"]["go"]["GO"], flush=True)


def write_summary(out, summary, checks):
    lines = [f"# M1 summary ({datetime.now(timezone.utc).isoformat()})", "",
             "Measured, layer 19, 5-fold CV by family on the 333 correction pairs.", "", "## VERIFY", ""]
    lines += [f"- {n}:{k}: {'pass' if v['pass'] else 'FAIL'}" for n, c in checks.items() for k, v in c.items()]
    for name in MODELS:
        go = summary[name]["go"]
        lines += ["", f"## {name}", "", f"Best: version {go['best']['version']}, lambda {go['best']['lambda']:g} "
                  f"(mean catch_inv at d=8 over hi-Latn, en-hi = {go['selection_score']:.3f})",
                  f"R hi-Latn = {go['R_hi-Latn']['R']:.3f}, fold CI [{go['R_hi-Latn']['fold_ci'][0]:.3f}, "
                  f"{go['R_hi-Latn']['fold_ci'][1]:.3f}] (info); pair-bootstrap CI "
                  f"[{go['R_hi-Latn_bootstrap']['ci'][0]:.3f}, {go['R_hi-Latn_bootstrap']['ci'][1]:.3f}] "
                  f"({go['R_hi-Latn_bootstrap']['n_undefined']} undefined resamples); R en-hi = {go['R_en-hi']['R']:.3f}; "
                  f"control R hi-Latn = {go['control_R_hi-Latn']:.3f}", ""]
        lines += [f"- {k}: {'pass' if v else 'fail'}" for k, v in go["rules"].items()]
        lines += [f"- **GO: {go['GO']}**" + (" (decision model)" if name == "read" else " (report only)"), "",
                  "| version | lambda | hi-Latn catch ctx/inv/iso d8 | en-hi catch ctx/inv/iso d8 | R hi-Latn | R en-hi | control R hi-Latn |",
                  "|---|---|---|---|---|---|---|"]
        t = summary[name]["table"]
        for v in VERSIONS:
            for lam in LAMBDAS:
                res, ctrl = t[f"{v}|{lam:g}|real"], t[f"{v}|{lam:g}|control"]
                cells = [" / ".join(f"{res['metrics'][n][f'{s}|8']['catch']:.3f}" for n in ("ctx", "inv", "iso"))
                         for s in ("hi-Latn", "en-hi")]
                lines.append(f"| {v} | {lam:g} | {cells[0]} | {cells[1]} | {res['R']['hi-Latn']['R']:.3f} | "
                             f"{res['R']['en-hi']['R']:.3f} | {ctrl['R']['hi-Latn']['R']:.3f} |")
        best = t[f"{go['best']['version']}|{go['best']['lambda']:g}|real"]["metrics"]
        lines += ["", "AUROC of the best configuration (ctx / inv / iso):", "", "| script | " +
                  " | ".join(f"d{d}" for d in DEPTHS) + " |", "|---" * 5 + "|"]
        for s in SCRIPTS:
            lines.append(f"| {s} | " + " | ".join(" / ".join(f"{best[n][f'{s}|{d}']['auroc']:.3f}" for n in ("ctx", "inv", "iso"))
                                                for d in DEPTHS) + " |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

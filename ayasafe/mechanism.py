"""Mechanism on the 333 correction pairs (older heretic set: prompts dataset data/heretic).
  cache       last-prompt-token residuals [P, 2 (benign, harmful), S, D, L, 4096], bf16 forward, fp32 storage
  r_hat       English refusal direction per layer: normalized mean(harmful) - mean(benign), single turn
  projections rho = share of a cell's mean harm difference NOT along the English depth-0 one (layer 19, 95% pair
              bootstrap 2000, seed 0); out-of-fold AUROC of the English direction per layer at depths 0 and 8
  ablation    generation with r_hat projected out after the embeddings and every decoder layer (English harmful prompts)
  probe       out-of-fold harmful-vs-benign AUROC of linear / RBF-SVM / MLP probes per (script, depth) at one layer
Layer index = hidden_states index (layer 19 = output of layers[18])."""
import json
from pathlib import Path

import numpy as np
import torch

from . import data, model as M, repair

SCRIPTS, DEPTHS = ("en", "hi-Deva", "hi-Latn", "en-hi"), (0, 2, 5, 8)
LAYERS = sorted(set(range(0, 31, 2)) | set(range(12, 20)))  # 0 = embedding output


def auroc(pos, neg) -> float:
    p, n = np.asarray(pos, float)[:, None], np.asarray(neg, float)[None, :]
    return float((p > n).mean() + 0.5 * (p == n).mean())


@torch.no_grad()
def cache(cfg):
    """-> <out>/h.pt and <out>/pairs.json. Batches per (script, depth), longest first."""
    out = Path(cfg["cache"])
    out.mkdir(parents=True, exist_ok=False)
    tok, model = M.load()
    pairs = sorted({r["pair_id"] for r in data.correction(cfg["correction"], "en", 0)})
    P = {p: i for i, p in enumerate(pairs)}
    h = torch.zeros(len(pairs), 2, len(SCRIPTS), len(DEPTHS), len(LAYERS), model.config.hidden_size)
    store = {}
    keep = lambda l: (lambda _m, _i, o: store.__setitem__(l, (o[0] if isinstance(o, tuple) else o)[:, -1]))  # noqa: E731
    hooks = [(model.model.embed_tokens if l == 0 else model.model.layers[l - 1]).register_forward_hook(keep(l)) for l in LAYERS]
    try:
        for s, sc in enumerate(SCRIPTS):
            for d, dep in enumerate(DEPTHS):
                rows = data.correction(cfg["correction"], sc, dep)
                enc = M.encode(tok, rows)
                order = sorted(range(len(rows)), key=lambda i: -len(enc[i]))
                for b in range(0, len(order), 16):
                    idx = order[b:b + 16]
                    model.model(**tok.pad({"input_ids": [enc[i] for i in idx]}, return_tensors="pt").to(model.device))
                    for j, i in enumerate(idx):
                        h[P[rows[i]["pair_id"]], int(rows[i]["request_type"] == "harmful"), s, d] = torch.stack([store[l][j].float() for l in LAYERS]).cpu()
    finally:
        for x in hooks:
            x.remove()
    torch.save(h, out / "h.pt")
    (out / "pairs.json").write_text(json.dumps(pairs))


def r_hat(h: torch.Tensor) -> dict:
    out = {}
    for s, sc in enumerate(SCRIPTS):
        diff = h[:, 1, s, 0].double().mean(0) - h[:, 0, s, 0].double().mean(0)
        out[sc] = (diff / diff.norm(dim=-1, keepdim=True).clamp(min=1e-12)).float()
    return {"layers": LAYERS, "r_hat": out}


def rho(Dh: torch.Tensor, r: torch.Tensor, n_boot: int = 2000, seed: int = 0) -> dict:
    """Dh [P, S, D, F] paired harm differences at one layer -> rho and cos(Delta, r_hat), point + 95% CI per cell."""
    P = Dh.shape[0]
    g = torch.Generator().manual_seed(seed)
    counts = torch.zeros(n_boot + 1, P, dtype=Dh.dtype)
    counts[0] = 1.0
    idx = torch.randint(0, P, (n_boot, P), generator=g)
    counts[1:].scatter_add_(1, idx, torch.ones_like(idx, dtype=Dh.dtype))
    mean = torch.einsum("bp,psdf->bsdf", counts / P, Dh)
    u = mean[:, SCRIPTS.index("en"), 0]
    u = u / u.norm(dim=1, keepdim=True)
    norm = mean.norm(dim=-1)
    along = torch.einsum("bsdf,bf->bsd", mean, u)
    vals = {"rho": (mean - along[..., None] * u[:, None, None]).norm(dim=-1) / norm, "cos_r_hat": torch.einsum("bsdf,f->bsd", mean, r) / norm}
    return {f"{sc}|{d}": {k: {"point": float(v[0, si, di]), "ci": torch.quantile(v[1:, si, di], torch.tensor([0.025, 0.975], dtype=v.dtype)).tolist()}
                          for k, v in vals.items()} for si, sc in enumerate(SCRIPTS) for di, d in enumerate(DEPTHS)}


def depth_loss(h: torch.Tensor, pairs: list) -> dict:
    """Per layer: English depth-0 direction fit out of fold (5 folds by pair); AUROC per script at depths 0 and 8."""
    f = repair.folds(pairs, 5, 0)
    fold = np.array([f[p] for p in pairs])
    res = {}
    for li, layer in enumerate(LAYERS):
        X = h[:, :, :, :, li].double()
        sc = torch.zeros(X.shape[:4], dtype=torch.float64)
        for k in range(5):
            tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
            r = X[tr, 1, 0, 0].mean(0) - X[tr, 0, 0, 0].mean(0)
            sc[te] = torch.einsum("pasdf,f->pasd", X[te], r / r.norm() if r.norm() > 0 else r)
        res[layer] = {f"{s}|{d}": auroc(sc[:, 1, SCRIPTS.index(s), DEPTHS.index(d)].tolist(), sc[:, 0, SCRIPTS.index(s), DEPTHS.index(d)].tolist())
                      for s in ("en", "hi-Deva", "hi-Latn") for d in (0, 8)}
    return res


def probe(Xc: np.ndarray, y: np.ndarray, groups: np.ndarray, fold_of: dict, kind: str, seed: int = 0) -> float:
    """Out-of-fold AUROC of one probe (linear: standardized L2 logistic, C by inner 3-fold GroupKFold; rbf: SVM, C by
    inner CV; mlp: 256 units, early stopping)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV, GroupKFold
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    score, outer = np.zeros(len(y)), np.array([fold_of[g] for g in groups])
    for f in sorted(set(outer)):
        tr, te = outer != f, outer == f
        if kind == "linear":
            m = GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)), {"logisticregression__C": [1e-3, 1e-2, 1e-1, 1.0]},
                             cv=GroupKFold(3), scoring="roc_auc").fit(Xc[tr], y[tr], groups=groups[tr])
        elif kind == "rbf":
            m = GridSearchCV(SVC(kernel="rbf", gamma="scale"), {"C": [0.1, 1.0, 10.0, 100.0]}, cv=GroupKFold(3), scoring="roc_auc").fit(Xc[tr], y[tr], groups=groups[tr])
        else:
            m = make_pipeline(StandardScaler(), MLPClassifier((256,), early_stopping=True, max_iter=500, random_state=seed)).fit(Xc[tr], y[tr])
        score[te] = m.predict_proba(Xc[te])[:, 1] if kind == "mlp" else m.decision_function(Xc[te])
    return auroc(score[y == 1], score[y == 0])


@torch.no_grad()
def ablation(cfg):
    """English harmful single-turn held-out prompts generated with r_hat (cfg layer) projected out everywhere; label the
    replies with judge.label_v3 / judge.label afterwards. Writes <out>/ablate-L<layer>.jsonl."""
    tok, model = M.load()
    rows = [r for r in data.correction(cfg["correction"], "en", 0, "heldout") if r["request_type"] == "harmful"]
    out = Path(cfg["out"])
    out.mkdir(parents=True, exist_ok=True)
    for layer in cfg["ablation_layers"]:
        r = torch.load(cfg["r_hat"])["r_hat"]["en"][layer, 0].to(model.device)

        def proj(_m, _i, o, r=r):
            h = o[0] if isinstance(o, tuple) else o
            h = h - (h @ r.to(h.dtype)).unsqueeze(-1) * r.to(h.dtype)
            return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h
        hooks = [m.register_forward_hook(proj) for m in (model.model.embed_tokens, *model.model.layers)]
        try:
            res = M.generate(tok, model, rows, 16, cfg["max_new_tokens"])
        finally:
            for x in hooks:
                x.remove()
        M.save(out / f"ablate-L{layer}.jsonl", rows, res, f"ablate-L{layer}", {"ablate_layer": layer}, cfg["max_new_tokens"])


def run(cfg):
    """cache (optional, GPU) -> r_hat, rho, depth loss, probes at cfg['probe_layer'] (CPU) -> <out>/mechanism.json; ablation (GPU, optional)."""
    if cfg.get("build_cache"):
        cache(cfg)
    src, out = Path(cfg["cache"]), Path(cfg["out"])
    out.mkdir(parents=True, exist_ok=True)
    h = torch.load(src / "h.pt", mmap=True)
    pairs = json.loads(Path(cfg["pairs"]).read_text())
    rh = r_hat(h)
    torch.save(rh, out / "r_hat.pt")
    li = LAYERS.index(M.LAYER)
    X = h[:, :, :, :, li].double()
    f = repair.folds(pairs, 5, 0)
    fold_of = {i: f[p] for i, p in enumerate(pairs)}
    probes = {}
    for s, sc in enumerate(SCRIPTS):
        for d, dep in enumerate(DEPTHS):
            Xc = h[:, :, s, d, LAYERS.index(cfg["probe_layer"])].double().reshape(-1, h.shape[-1]).numpy()
            y, g = np.tile([0, 1], h.shape[0]), np.repeat(np.arange(h.shape[0]), 2)
            probes[f"{sc}|{dep}"] = {k: probe(Xc, y, g, fold_of, k) for k in ("linear", "rbf", "mlp")}
    res = {"rho_layer19": rho(X[:, 1] - X[:, 0], rh["r_hat"]["en"][li].double()), "depth_loss": depth_loss(h, pairs), "probes": probes}
    (out / "mechanism.json").write_text(json.dumps(res, indent=1))
    if cfg.get("ablation_layers"):
        ablation(cfg)


# ------------------------------------------------------------------ alignment cosine (docs/prereg_alignment.md)
def _cos(a, b):
    return float(a @ b / (a.norm() * b.norm()))


@torch.no_grad()
def alignment_acts(cfg):
    """Forward only: fp32 hidden_states[19] at the last prompt token for every fresh TEST row per (condition, depth)."""
    from . import normalize
    out = Path(cfg["out"])
    out.mkdir(parents=True, exist_ok=True)
    tok, model = M.load()
    auto = normalize.load(cfg["xlit"])
    acts = {}
    for cond, script in (("en", "en"), ("hi-Deva", "hi-Deva"), ("hi-Latn", "hi-Latn"), ("hi-Latn+N", "hi-Latn"), ("en-hi", "en-hi")):
        for d in (0, 8):
            rows = data.rows(Path(cfg["conversations"]), script, d, "test")
            if cond == "hi-Latn+N":
                normalize.apply(rows, auto)
            H = M.hidden(tok, model, rows, bs=cfg.get("batch_size", 16))
            acts[f"{cond}|{d}"] = {"h": H, "group_id": [r["group_id"] for r in rows], "harm": [r["request_type"] == "harmful" for r in rows]}
    torch.save(acts, out / "acts.pt")


def alignment(cfg):
    """alignment.csv from the stored activations: d, cosines with r_hat / w_new / w_old / d_en, gap, half-split ceiling,
    pair-bootstrap CIs (4000, seed 0), P-A..P-D."""
    import pandas as pd
    out = Path(cfg["out"])
    acts = torch.load(out / "acts.pt")
    r = repair.r_hat(cfg["r_hat"]).double()
    w_new, w_old = torch.load(cfg["w_new"])["w_base"].double(), torch.load(cfg["w_old"])["w_base"].double()
    D = {}
    for k, a in acts.items():
        by = {}
        for h, g, harm in zip(a["h"].double(), a["group_id"], a["harm"]):
            by.setdefault(g, [None, None])[int(harm)] = h
        gs = sorted(g for g, v in by.items() if None not in v)
        D[k] = torch.stack([by[g][1] - by[g][0] for g in gs])  # [pairs, F]
    rng = np.random.default_rng(0)
    rows = []
    for k, X in D.items():
        cond, d = k.split("|")
        en = D[f"en|{d}"]
        P = X.shape[0]

        def stats(idx):
            m = X[idx].mean(0)
            me = en[idx].mean(0) if en.shape[0] == P else en.mean(0)
            return {"cos_r": _cos(m, r), "cos_wnew": _cos(m, w_new), "cos_wold": _cos(m, w_old), "cos_en": _cos(m, me), "gap_r": float(r @ m)}
        point = stats(np.arange(P))
        boots = [stats(rng.integers(0, P, P)) for _ in range(cfg.get("n_boot", 4000))]
        halves = []
        for _ in range(200):
            p = rng.permutation(P)
            halves.append(_cos(X[p[: P // 2]].mean(0), X[p[P // 2:]].mean(0)))
        rec = {"condition": cond, "depth": int(d), "n": P}
        for m, v in point.items():
            b = np.array([x[m] for x in boots])
            rec.update({m: v, f"{m}_ci_low": np.quantile(b, 0.025), f"{m}_ci_high": np.quantile(b, 0.975)})
        rec.update({"ceiling": float(np.mean(halves)), "ceiling_ci_low": float(np.quantile(halves, 0.025)), "ceiling_ci_high": float(np.quantile(halves, 0.975))})
        rows.append(rec)
    t = pd.DataFrame(rows)
    g = lambda c, d, m="cos_r": t[(t.condition == c) & (t.depth == d)].iloc[0][m]  # noqa: E731
    res = {}
    for d in (0, 8):
        res[f"P-A d{d}"] = g("en", d) > g("hi-Deva", d) > g("hi-Latn", d)
        res[f"P-B d{d}"] = g("hi-Deva", d, "cos_r_ci_low") <= g("hi-Latn+N", d) <= g("hi-Deva", d, "cos_r_ci_high")
        res[f"P-C d{d}"] = g("hi-Latn+N", d, "cos_wnew_ci_low") > g("hi-Latn+N", d, "cos_wold_ci_high")
    res["P-D"] = g("hi-Latn", 8) < g("hi-Latn", 0)
    t["predictions"] = json.dumps({k: bool(v) for k, v in res.items()})
    t["source"] = "ayasafe.mechanism.alignment; docs/prereg_alignment.md"
    t.to_csv(out / "alignment.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    conds = ["en", "hi-Deva", "hi-Latn", "hi-Latn+N", "en-hi"]
    for j, d in enumerate((0, 8)):
        s = t[t.depth == d].set_index("condition").loc[conds]
        x = np.arange(len(conds)) + (j - 0.5) * 0.38
        ax[0].bar(x, s.cos_r, 0.36, yerr=[s.cos_r - s.cos_r_ci_low, s.cos_r_ci_high - s.cos_r], label=f"depth {d}", capsize=3)
        ax[0].hlines(s.ceiling, x - 0.18, x + 0.18, colors="k", linestyles="dashed")
        n = s.loc["hi-Latn+N"]
        ax[1].bar([j - 0.2, j + 0.2], [n.cos_wnew, n.cos_wold], 0.36, color=["C2", "C3"],
                  yerr=[[n.cos_wnew - n.cos_wnew_ci_low, n.cos_wold - n.cos_wold_ci_low], [n.cos_wnew_ci_high - n.cos_wnew, n.cos_wold_ci_high - n.cos_wold]], capsize=3)
    ax[0].set_xticks(range(len(conds)), conds)
    ax[0].set_ylabel("cos(d, r_hat), 95% CI; dashed = half-split ceiling")
    ax[0].legend()
    ax[1].set_xticks([-0.2, 0.2, 0.8, 1.2], ["w_new d0", "w_old d0", "w_new d8", "w_old d8"])
    ax[1].set_ylabel("cos(d, w), hi-Latn after N")
    fig.tight_layout()
    fig.savefig(out / "alignment.png", dpi=130)
    return res


def alignment_run(cfg):
    if not (Path(cfg["out"]) / "acts.pt").exists():
        alignment_acts(cfg)
    print(alignment(cfg))


def separation(cfg):
    """ADDED ANALYSIS (not pre-registered). Per condition and depth, for v in {w_new, w_old, r_hat}: paired d' =
    mean(v.(h_harm - h_ben)) / sd(v.(h_harm - h_ben)) (= v.dbar / sqrt(v'Sv), S = covariance of paired differences) and
    AUROC of v.h (harmful vs benign); 95% percentile bootstrap over pairs (4000, seed 0). Appends columns to alignment.csv
    and redraws alignment.png with a third panel."""
    import pandas as pd
    out = Path(cfg["out"])
    acts = torch.load(out / "acts.pt")
    V = {"w_new": torch.load(cfg["w_new"])["w_base"].double(), "w_old": torch.load(cfg["w_old"])["w_base"].double(),
         "r_hat": repair.r_hat(cfg["r_hat"]).double()}
    if (out / "w_raw_fresh.pt").exists():  # from raw_fit
        V["w_raw_fresh"] = torch.load(out / "w_raw_fresh.pt")["w_raw_fresh"].double()
    t = pd.read_csv(out / "alignment.csv")
    rng = np.random.default_rng(0)
    for k, a in acts.items():
        cond, d = k.split("|")
        by = {}
        for h, g, harm in zip(a["h"].double(), a["group_id"], a["harm"]):
            by.setdefault(g, [None, None])[int(harm)] = h
        gs = sorted(g for g, v in by.items() if None not in v)
        Hh, Hb = torch.stack([by[g][1] for g in gs]), torch.stack([by[g][0] for g in gs])
        idx = rng.integers(0, len(gs), (cfg.get("n_boot", 4000), len(gs)))
        i = (t.condition == cond) & (t.depth == int(d))
        for name, v in V.items():
            ph, pb = (Hh @ v).numpy(), (Hb @ v).numpy()
            diff = ph - pb
            dp = lambda s: diff[s].mean() / diff[s].std(ddof=1)  # noqa: E731
            au = lambda s: auroc(ph[s], pb[s])  # noqa: E731
            bd = np.array([dp(s) for s in idx])
            ba = np.array([au(s) for s in idx])
            t.loc[i, [f"dprime_{name}", f"dprime_{name}_ci_low", f"dprime_{name}_ci_high"]] = [dp(slice(None)), *np.quantile(bd, [0.025, 0.975])]
            t.loc[i, [f"auroc_{name}", f"auroc_{name}_ci_low", f"auroc_{name}_ci_high"]] = [au(slice(None)), *np.quantile(ba, [0.025, 0.975])]
    t["added_analysis"] = "dprime_* and auroc_*: added after the pre-registration (not pre-registered)"
    t.to_csv(out / "alignment.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    conds = ["en", "hi-Deva", "hi-Latn", "hi-Latn+N", "en-hi"]
    for j, d in enumerate((0, 8)):
        s = t[t.depth == d].set_index("condition").loc[conds]
        x = np.arange(len(conds)) + (j - 0.5) * 0.38
        ax[0].bar(x, s.cos_r, 0.36, yerr=[(s.cos_r - s.cos_r_ci_low).clip(lower=0), (s.cos_r_ci_high - s.cos_r).clip(lower=0)], label=f"depth {d}", capsize=3)
        ax[0].hlines(s.ceiling, x - 0.18, x + 0.18, colors="k", linestyles="dashed")
        n = s.loc["hi-Latn+N"]
        ax[1].bar([j - 0.2, j + 0.2], [n.cos_wnew, n.cos_wold], 0.36, color=["C2", "C3"], capsize=3,
                  yerr=[[max(0, n.cos_wnew - n.cos_wnew_ci_low), max(0, n.cos_wold - n.cos_wold_ci_low)],
                        [max(0, n.cos_wnew_ci_high - n.cos_wnew), max(0, n.cos_wold_ci_high - n.cos_wold)]])
        names = [x for x in (("w_new", "C2"), ("w_raw_fresh", "C4"), ("w_old", "C3"), ("r_hat", "C0")) if x[0] in V]
        for q, (name, c) in enumerate(names):
            xs = np.arange(len(conds)) * 2 + j * 0.9 + (q - (len(names) - 1) / 2) * 0.2
            m = s[f"dprime_{name}"]
            ax[2].bar(xs, m, 0.19, color=c, alpha=1 if d == 0 else 0.55, capsize=2, label=f"{name} d{d}",
                      yerr=[(m - s[f"dprime_{name}_ci_low"]).clip(lower=0), (s[f"dprime_{name}_ci_high"] - m).clip(lower=0)])
    ax[0].set_xticks(range(len(conds)), conds)
    ax[0].set_ylabel("cos(d, r_hat), 95% CI; dashed = half-split ceiling")
    ax[0].legend()
    ax[1].set_xticks([-0.2, 0.2, 0.8, 1.2], ["w_new d0", "w_old d0", "w_new d8", "w_old d8"])
    ax[1].set_ylabel("cos(d, w), hi-Latn after N")
    ax[2].set_xticks(np.arange(len(conds)) * 2 + 0.45, conds)
    ax[2].set_ylabel("paired d' (added analysis), 95% CI")
    ax[2].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "alignment.png", dpi=130)


@torch.no_grad()
def raw_fit(cfg):
    """ADDED ANALYSIS. w_raw_fresh: the w_new recipe (repair.fisher_w; base weights; layer 19 last token; scripts
    hi-Latn, en-hi, en, hi-Deva x depths 0/2/5/8; fit on calibration families of fold 0, folds(k=2, seed 0)) but with RAW
    hi-Latn. The same fit on normalized hi-Latn is a check against the stored w_new. Then on the test activations:
    d' / AUROC of w_raw_fresh, and paired-bootstrap d' contrasts (same resampled pairs for both vectors)."""
    import pandas as pd
    from . import normalize
    out, conv = Path(cfg["out"]), Path(cfg["conversations"])
    scripts, depths = ("hi-Latn", "en-hi", "en", "hi-Deva"), (0, 2, 5, 8)
    f = out / "cal_acts.pt"
    if not f.exists():
        tok, model = M.load()
        auto = normalize.load(cfg["xlit_calibration"])
        acts = {}
        for sc in scripts + ("hi-Latn+N",):
            for d in depths:
                rows = data.rows(conv, "hi-Latn" if sc == "hi-Latn+N" else sc, d, "calibration")
                if sc == "hi-Latn+N":
                    normalize.apply(rows, auto)
                acts[f"{sc}|{d}"] = {"h": M.hidden(tok, model, rows, bs=cfg.get("batch_size", 16)), "pair_id": [r["pair_id"] for r in rows],
                                     "family_id": [r["family_id"] for r in rows], "harm": [r["request_type"] == "harmful" for r in rows]}
        torch.save(acts, f)
        del model
    acts = torch.load(f)
    a0 = acts["hi-Latn|0"]
    pairs = sorted(set(a0["pair_id"]))
    fam = dict(zip(a0["pair_id"], a0["family_id"]))
    fold = repair.folds([fam[p] for p in pairs], 2, 0)
    fit_idx = np.array([i for i, p in enumerate(pairs) if fold[fam[p]] == 0])

    def X_of(first):
        X = torch.zeros(len(pairs), 2, len(scripts), len(depths), 4096, dtype=torch.float64)
        for s, sc in enumerate((first,) + scripts[1:]):
            for d, dep in enumerate(depths):
                a = acts[f"{sc}|{dep}"]
                for h, p, harm in zip(a["h"], a["pair_id"], a["harm"]):
                    X[pairs.index(p), int(harm), s, d] = h.double()
        return X
    w_raw = repair.fisher_w(X_of("hi-Latn"), fit_idx)
    w_chk = repair.fisher_w(X_of("hi-Latn+N"), fit_idx)
    w_new = torch.load(cfg["w_new"])["w_base"].double()
    w_old = torch.load(cfg["w_old"])["w_base"].double()
    torch.save({"w_raw_fresh": w_raw.float(), "w_new_refit_check": w_chk.float()}, out / "w_raw_fresh.pt")
    check = {"cos(w_new_refit, w_new_stored)": float(w_chk @ w_new / w_new.norm()), "cos(w_raw_fresh, w_new)": float(w_raw @ w_new / w_new.norm()),
             "cos(w_raw_fresh, w_old)": float(w_raw @ w_old / w_old.norm())}
    test = torch.load(out / "acts.pt")
    t = pd.read_csv(out / "alignment.csv")
    rng = np.random.default_rng(0)
    V = {"w_raw_fresh": w_raw, "w_new": w_new, "w_old": w_old}
    contrasts = []
    for k, a in test.items():
        cond, d = k.split("|")
        by = {}
        for h, g, harm in zip(a["h"].double(), a["group_id"], a["harm"]):
            by.setdefault(g, [None, None])[int(harm)] = h
        gs = sorted(g for g, v in by.items() if None not in v)
        Hh, Hb = torch.stack([by[g][1] for g in gs]), torch.stack([by[g][0] for g in gs])
        idx = rng.integers(0, len(gs), (cfg.get("n_boot", 4000), len(gs)))
        P = {n: ((Hh @ v).numpy(), (Hb @ v).numpy()) for n, v in V.items()}
        dp = lambda n, s: (lambda x: x.mean() / x.std(ddof=1))(P[n][0][s] - P[n][1][s])  # noqa: E731
        i = (t.condition == cond) & (t.depth == int(d))
        bd = np.array([dp("w_raw_fresh", s) for s in idx])
        ba = np.array([auroc(P["w_raw_fresh"][0][s], P["w_raw_fresh"][1][s]) for s in idx])
        t.loc[i, ["dprime_w_raw_fresh", "dprime_w_raw_fresh_ci_low", "dprime_w_raw_fresh_ci_high"]] = [dp("w_raw_fresh", slice(None)), *np.quantile(bd, [0.025, 0.975])]
        t.loc[i, ["auroc_w_raw_fresh", "auroc_w_raw_fresh_ci_low", "auroc_w_raw_fresh_ci_high"]] = [auroc(*P["w_raw_fresh"]), *np.quantile(ba, [0.025, 0.975])]
        for name, a_, b_ in (("data effect: w_raw_fresh - w_old", "w_raw_fresh", "w_old"), ("space effect: w_new - w_raw_fresh", "w_new", "w_raw_fresh")):
            if cond in ("hi-Latn", "hi-Latn+N"):
                diff = np.array([dp(a_, s) - dp(b_, s) for s in idx])
                contrasts.append({"contrast": name, "condition": cond, "depth": int(d), "delta_dprime": dp(a_, slice(None)) - dp(b_, slice(None)),
                                  "ci_low": np.quantile(diff, 0.025), "ci_high": np.quantile(diff, 0.975), "n": len(gs)})
    t["added_analysis"] = "dprime_*, auroc_* (incl. w_raw_fresh): added after the pre-registration (not pre-registered)"
    t.to_csv(out / "alignment.csv", index=False)
    c = pd.DataFrame(contrasts)
    c["added_analysis"] = True
    c["check"] = json.dumps(check)
    c.to_csv(out / "alignment_contrasts.csv", index=False)
    print(json.dumps(check), "\n", c.round(3).to_string(index=False))


# ------------------------------------------------------------------ held-out split activations (exploration-set paper)
@torch.no_grad()
def heldout_acts(cfg):
    """Forward only: fp32 hidden_states[19], last prompt token. Held-out split: en, hi-Deva, hi-Latn, en-hi, hi-Latn
    transliterated (gold) at depth 0 / 5 (pilot prefixes) / 8 (main prefixes), as in the stored held-out runs. Fitting
    split: transliterated hi-Latn at depths 0/2/5/8 (main prefixes) for the w_tr fit. -> <out>/heldout_acts.pt, fit_tr_acts.pt"""
    out = Path(cfg["out"])
    out.mkdir(parents=True, exist_ok=True)
    tok, model = M.load()
    auto = {x["item_id"]: x["auto"] for x in data.jl(cfg["xlit"])}
    pre = cfg["prefixes"]

    def rows(script, depth, split, pool):
        if split == "heldout":  # exactly the conversations of the stored held-out runs (Base; Transliteration for "tr")
            f = Path(cfg["heldout_runs"][depth]) / f"{'hi-Latn-N' if script == 'tr' else script + '-Zb'}.jsonl"
            return [{**x, "key": x["key"]} for x in data.jl(f)]
        r = data.correction({"heretic": cfg["heretic"][pool], "prefixes": pre}, "hi-Latn" if script == "tr" else script, depth, split)
        if script == "tr":
            for x in r:
                x["model_request"] = auto[x["item_id"]]
        return r
    bs = cfg.get("batch_size", 4)
    if not (out / "heldout_acts.pt").exists():
        acts = {}
        for sc in ("en", "hi-Deva", "hi-Latn", "en-hi", "tr"):
            for d, pool in ((0, "main"), (5, "pilot"), (8, "main")):
                r = rows(sc, d, "heldout", pool)
                acts[f"{sc}|{d}"] = {"h": M.hidden(tok, model, r, bs=bs), "pair_id": [x["pair_id"] for x in r], "harm": [x["request_type"] == "harmful" for x in r]}
        torch.save(acts, out / "heldout_acts.pt")
    if not (out / "fit_tr_acts.pt").exists():
        acts = {}
        for d in (0, 2, 5, 8):
            r = rows("tr", d, "correction", "main")
            acts[d] = {"h": M.hidden(tok, model, r, bs=bs), "pair_id": [x["pair_id"] for x in r], "harm": [x["request_type"] == "harmful" for x in r]}
        torch.save(acts, out / "fit_tr_acts.pt")


def heldout_mech(cfg):
    """CPU, from heldout_acts.pt / fit_tr_acts.pt: per (condition, depth) on the held-out split: cos(d, r_hat) with half-
    split ceiling; d' / AUROC for r_hat, w (fitting split, raw romanized; stored calib_w direction) and w_tr (same recipe
    with transliterated romanized rows); space effect d'(w_tr) - d'(w), paired bootstrap (4000, seed 0)."""
    import pandas as pd
    out = Path(cfg["out"])
    A = torch.load(out / "heldout_acts.pt")
    r = repair.r_hat(cfg["r_hat"]).double()
    w = torch.load(cfg["w"])["w_base"].double()
    # w_tr: M0 cache (fitting split, base) with the hi-Latn slot replaced by transliterated activations; same pairs as w
    pairs = json.loads(Path(cfg["cache_pairs"]).read_text())
    X = torch.load(cfg["cache"], mmap=True)[..., LAYERS.index(M.LAYER), :].double().clone()
    T = torch.load(out / "fit_tr_acts.pt")
    P = {p: i for i, p in enumerate(pairs)}
    for di, d in enumerate(DEPTHS):
        for h, p, harm in zip(T[d]["h"], T[d]["pair_id"], T[d]["harm"]):
            X[P[p], int(harm), SCRIPTS.index("hi-Latn"), di] = h.double()
    fit_idx = np.array([i for i, p in enumerate(pairs) if p in set(json.loads(Path(cfg["fit_pairs"]).read_text()))])
    w_tr = repair.fisher_w(X, fit_idx)
    w_chk = repair.fisher_w(torch.load(cfg["cache"], mmap=True)[..., LAYERS.index(M.LAYER), :].double(), fit_idx)
    rng = np.random.default_rng(0)
    rows = []
    for k, a in A.items():
        cond, d = k.split("|")
        by = {}
        for h, p, harm in zip(a["h"].double(), a["pair_id"], a["harm"]):
            by.setdefault(p, [None, None])[int(harm)] = h
        ids = sorted(p for p, v in by.items() if None not in v)
        Hh, Hb = torch.stack([by[p][1] for p in ids]), torch.stack([by[p][0] for p in ids])
        D = Hh - Hb
        idx = rng.integers(0, len(ids), (cfg.get("n_boot", 4000), len(ids)))
        cosr = lambda s: float(D[s].mean(0) @ r / D[s].mean(0).norm())  # noqa: E731
        half = []
        for _ in range(200):
            q = rng.permutation(len(ids))
            m1_, m2_ = D[q[: len(ids) // 2]].mean(0), D[q[len(ids) // 2:]].mean(0)
            half.append(float(m1_ @ m2_ / (m1_.norm() * m2_.norm())))
        bc = np.array([cosr(s) for s in idx])
        rec = {"condition": cond, "depth": int(d), "n_pairs": len(ids), "cos_r": cosr(slice(None)), "cos_r_ci_low": np.quantile(bc, 0.025),
               "cos_r_ci_high": np.quantile(bc, 0.975), "ceiling": float(np.mean(half)), "proj_r": float((D @ r).mean())}
        proj = {n: ((Hh @ v).numpy(), (Hb @ v).numpy()) for n, v in (("r_hat", r), ("w", w), ("w_tr", w_tr))}
        dp = lambda n, s: (lambda x: x.mean() / x.std(ddof=1))(proj[n][0][s] - proj[n][1][s])  # noqa: E731
        for n in proj:
            b = np.array([dp(n, s) for s in idx])
            ba = np.array([auroc(proj[n][0][s], proj[n][1][s]) for s in idx])
            rec.update({f"dprime_{n}": dp(n, slice(None)), f"dprime_{n}_ci_low": np.quantile(b, 0.025), f"dprime_{n}_ci_high": np.quantile(b, 0.975),
                        f"auroc_{n}": auroc(*proj[n]), f"auroc_{n}_ci_low": np.quantile(ba, 0.025), f"auroc_{n}_ci_high": np.quantile(ba, 0.975)})
        sp = np.array([dp("w_tr", s) - dp("w", s) for s in idx])
        rec.update({"space_effect": dp("w_tr", slice(None)) - dp("w", slice(None)), "space_effect_ci_low": np.quantile(sp, 0.025),
                    "space_effect_ci_high": np.quantile(sp, 0.975)})
        rows.append(rec)
    t = pd.DataFrame(rows)
    t["check"] = json.dumps({"cos(refit w, stored w)": float(w_chk @ w / w.norm()), "cos(w_tr, w)": float(w_tr @ w / w.norm())})
    t.to_csv(out / "heldout_mech.csv", index=False)
    torch.save({"w_tr": w_tr.float()}, out / "w_tr.pt")
    print(t.check.iloc[0])

"""Repairs used in Step 5 and their fits.
  Read edit : closed-form soft edit of W_down at layer 16 (mu 1, fp64 solve) mapping romanized hi-Latn residuals at
              layer 17 toward the gold-Devanagari ones (final-turn mean + last token), fitted on the 333 correction pairs.
  w         : paired Fisher direction (version b: (S_pair + eps I)^-1 dbar, Ledoit-Wolf eps, fp64) at layer 19.
  gate G    : r_hat span gate, beta 16, threshold c_r per (script, depth) (split conformal, alpha 0.05).
  steer Sw  : w detects (threshold c_w), the push is a fixed beta * g_bar * r_hat on the final user turn.
  Rw        : Read-edited weights + Sw. N+X: normalization first, X refit on normalized calibration rows.
Fits use the fresh CALIBRATION split only."""
import json
import math
from pathlib import Path
import random

import numpy as np
import torch

from . import data, model as M, normalize


# ------------------------------------------------------------------ split-conformal threshold
def conformal(scores: list, alpha: float = 0.05) -> dict:
    """c = s_(k), k = ceil((n+1)(1-alpha)) over benign calibration scores (+inf if k > n). The gate fires when score > c."""
    n = len(scores)
    k = math.ceil((n + 1) * (1 - alpha))
    c = sorted(scores)[k - 1] if k <= n else float("inf")
    return {"c": c, "n": n, "k": k, "alpha": alpha, "calibration_fire_rate": sum(x > c for x in scores) / n}


# ------------------------------------------------------------------ paired Fisher w
def _lw(scatter, n, quartic):
    """Ledoit-Wolf shrinkage (closed form, assume_centered) from the centred scatter and sum ||x - m||^4."""
    p = scatter.shape[0]
    emp = scatter / n
    mu = float(torch.trace(emp)) / p
    d2 = float((emp ** 2).sum())
    beta = min((quartic / n - d2) / (p * n), (d2 - mu ** 2 * p) / p)
    delta = (d2 - mu ** 2 * p) / p
    return 0.0 if beta == 0 else beta / delta


def fisher_w(X: torch.Tensor, idx) -> torch.Tensor:
    """X [P, 2 (benign, harmful), S, D, F] fp64; fit on pairs idx. Unit w, harmful side positive."""
    d = (X[idx, 1] - X[idx, 0]).reshape(-1, X.shape[-1])
    m = d.mean(0)
    scat = (d - m).T @ (d - m)
    shrink = _lw(scat, d.shape[0], float((((d - m) ** 2).sum(1) ** 2).sum()))
    S = scat / (d.shape[0] - 1)
    S = S / (torch.trace(S) / S.shape[0])
    w = torch.linalg.solve(S + shrink / (1 - shrink) * torch.eye(S.shape[0], dtype=S.dtype), m)
    w = w / w.norm()
    return -w if float(torch.einsum("psdf,f->", X[idx, 1] - X[idx, 0], w)) < 0 else w


def folds(ids: list, k: int, seed: int = 0) -> dict:
    ids = sorted(set(ids))
    random.Random(seed).shuffle(ids)
    return {p: i % k for i, p in enumerate(ids)}


def r_hat(path: Path, layer: int = M.LAYER) -> torch.Tensor:
    """English r_hat at `layer` from mechanism.analyse output, or from the released artifact ([33, 2, 4096], position 0 = last token)."""
    d = torch.load(path)
    return d["r_hat"]["en"][d["layers"].index(layer)].float() if "layers" in d else d["r_hat"]["en"][layer, 0].float()


# ------------------------------------------------------------------ Step 5 fit
def fit(cfg):
    """w on half of the calibration families (fold 0), c_w on the other half's benign scores, c_r on all calibration
    benign scores, per (script, depth in gate_depths); g_bar = mean(r_hat.h - c_r) over fired calibration rows at depth 8."""
    out = Path(cfg["out"])
    out.mkdir(parents=True, exist_ok=False)
    conv, auto = Path(cfg["conversations"]), normalize.load(cfg["xlit_calibration"])
    scripts, depths = cfg["scripts"], cfg["depths"]
    tok, model = M.load()
    r = r_hat(cfg["r_hat"]).double()
    edit = M.WeightEdit(model, cfg["read_delta"])
    rows0 = data.rows(conv, "hi-Latn", 0, "calibration")
    pairs = sorted({x["pair_id"] for x in rows0})
    fam = {x["pair_id"]: x["family_id"] for x in rows0}
    fold = folds([fam[p] for p in pairs], 2, 0)
    fit_idx = np.array([i for i, p in enumerate(pairs) if fold[fam[p]] == 0])
    cal_idx = np.array([i for i, p in enumerate(pairs) if fold[fam[p]] == 1])
    res, dirs = {"pairs": len(pairs), "fit_pairs": len(fit_idx), "threshold_pairs": len(cal_idx)}, {}
    for name in ("base", "read"):
        if name == "read":
            edit.apply(1.0)
        X = torch.zeros(len(pairs), 2, len(scripts), len(depths), model.config.hidden_size)
        for s, sc in enumerate(scripts):
            for d, dep in enumerate(depths):
                rows = data.rows(conv, sc, dep, "calibration")
                if sc == "hi-Latn":
                    normalize.apply(rows, auto)
                for row, h in zip(rows, M.hidden(tok, model, rows)):
                    X[pairs.index(row["pair_id"]), int(row["request_type"] == "harmful"), s, d] = h
        X = X.double()
        w = fisher_w(X, fit_idx)
        sw, sr = torch.einsum("pasdf,f->pasd", X, w), torch.einsum("pasdf,f->pasd", X, r)
        cells = {}
        for s, sc in enumerate(scripts):
            for dep in cfg["gate_depths"]:
                d = depths.index(dep)
                cells[f"{sc}|{dep}"] = {"w": conformal(sw[cal_idx, 0, s, d].tolist(), cfg["alpha"]),
                                        "r_hat": conformal(sr[:, 0, s, d].tolist(), cfg["alpha"])}
        gbar = {}
        for s, sc in enumerate(scripts[:2]):
            c = cells[f"{sc}|8"]["r_hat"]["c"]
            v = sr[:, :, s, depths.index(8)].reshape(-1)
            gbar[sc] = {"g_bar": float((v[v > c] - c).mean()), "n_fired": int((v > c).sum())}
        for sc in scripts[2:]:
            gbar[sc] = {**gbar[scripts[0]], "from": scripts[0]}
        res[name] = {"thresholds": cells, "g_bar": gbar}
        dirs[f"w_{name}"] = w.float()
    edit.revert()
    torch.save(dirs | {"layer": M.LAYER}, out / "directions.pt")
    (out / "thresholds.json").write_text(json.dumps(res, indent=1))


# ------------------------------------------------------------------ generation (all arms)
def arm_setup(arm, script, depth, rows, fit_dir, old):
    """Per-row gate threshold and hook arguments for an arm; returns (weights, score_vec_key, c, g_push) or None."""
    if arm in ("Zb", "N", "Zl"):
        return None
    th = json.loads((fit_dir / "thresholds.json").read_text())
    weights = "read" if "Rw" in arm else "base"
    if arm == "N+G":
        return "base", "r_hat", th["base"]["thresholds"][f"{script}|{depth}"]["r_hat"]["c"], None
    if arm.endswith("_old"):
        o = json.loads(Path(old["thresholds"]).read_text())
        return weights, "w_old", o["thresholds"][weights][f"{script}|{depth}"]["w"]["c"], o["g_bar"][f"{weights}|{script}"]["g_bar"]
    return weights, "w", th[weights]["thresholds"][f"{script}|{depth}"]["w"]["c"], th[weights]["g_bar"][script]["g_bar"]


def generate(cfg):
    """cfg: conversations, split, runs [script:arm:depth], xlit, fit, old, r_hat, read_delta, prefix_latn, out, beta.
    Arms: Zb (as written), N (normalized), Zl (romanized prefix), N+G, N+Sw, N+Rw, N+Sw_old, N+Rw_old.
    Base-weight arms run before Read-weight arms; hi-Latn batches are fixed by the romanized prompts."""
    out, conv, fit_dir = Path(cfg["out"]), Path(cfg["conversations"]), Path(cfg["fit"])
    out.mkdir(parents=True, exist_ok=True)
    runs = [(s, a, int(d)) for s, a, d in (x.split(":") for x in cfg["runs"])]
    runs = [x for x in runs if "Rw" not in x[1]] + [x for x in runs if "Rw" in x[1]]
    tok, model = M.load()
    r = r_hat(cfg["r_hat"]).to(model.device)
    vecs = {"r_hat": r}
    for k, p in (("w", fit_dir / "directions.pt"), ("w_old", cfg["old"]["directions"])):
        dd = torch.load(p)
        vecs[k] = {m: dd[f"w_{m}"].float().to(model.device) for m in ("base", "read")}
    edit = M.WeightEdit(model, cfg["read_delta"])
    auto = normalize.load(cfg["xlit"])
    for s, a, d in runs:
        torch.manual_seed(42)
        rows = data.rows(conv, s, d, cfg["split"])
        if a == "Zl":
            pref = json.loads(Path(cfg["prefix_latn"]).read_text())
            for x in rows:
                x["opener_turns"] = pref[json.dumps(x["opener_turns"], ensure_ascii=False)]["latn"]
        order = M.batch_order(tok, rows) if s == "hi-Latn" else None
        if a.startswith("N") and s == "hi-Latn":
            normalize.apply(rows, auto)
        setup = arm_setup(a, s, d, rows, fit_dir, cfg.get("old"))
        weights = setup[0] if setup else "base"
        if weights == "read" and not edit.on:
            edit.apply(1.0)
        elif weights == "base" and edit.on:
            edit.revert()
        state, hooks = None, []
        if setup:
            _, vk, c, g_push = setup
            score = vecs[vk] if vk == "r_hat" else vecs[vk][weights]
            for x in rows:
                x["gate_c"] = c
            state = {}
            hooks = [M.span_gate(model, score, r, cfg["beta"], state, g_push)]
        try:
            res = M.generate(tok, model, rows, 16, cfg["max_new_tokens"], state, order)
        finally:
            for h in hooks:
                h.remove()
        M.save(out / f"{s}-{a}-d{d}.jsonl", rows, res, a, {"arm": a, "weights": weights, "normalized": a.startswith("N") and s == "hi-Latn",
                                                           "gate": setup[1:] if setup else None}, cfg["max_new_tokens"])
    edit.revert()


# ------------------------------------------------------------------ Read edit fit (layer 16, soft, all correction pairs)
@torch.no_grad()
def read_edit(cfg):
    """Delta = R Om K1^T (K1 Om K1^T + mu K0 K0^T + lam I)^-1 in fp64, lam = 1e-2 tr(A)/D. Targets: per item the
    final-turn mean and last-token residual differences (gold Devanagari - romanized) after layers[16]; preserve keys K0:
    24 random tokens per English single-turn, Devanagari single-turn + base answer, Devanagari depth-8 prompt."""
    src, L = cfg["correction"], cfg["layer"]
    tok, model = M.load()
    store = {}
    hk = [model.model.layers[L].mlp.down_proj.register_forward_hook(lambda _m, i, _o: store.__setitem__("k", i[0])),
          model.model.layers[L].register_forward_hook(lambda _m, _i, o: store.__setitem__("h", o[0] if isinstance(o, tuple) else o))]
    D = model.config.intermediate_size
    A = torch.zeros(D, D, dtype=torch.float64, device=model.device)
    C = torch.zeros(D, D, dtype=torch.float64, device=model.device)
    vec = {}
    eot = tok.convert_tokens_to_ids("<|END_OF_TURN_TOKEN|>")

    def run(seqs):
        order = sorted(range(len(seqs)), key=lambda i: -len(seqs[i]))
        for s in range(0, len(order), 16):
            idx = order[s:s + 16]
            batch = tok.pad({"input_ids": [seqs[i] for i in idx]}, return_tensors="pt").to(model.device)
            model.model(**batch)
            w = batch["input_ids"].shape[1]
            yield [(i, w - len(seqs[i]), j) for j, i in enumerate(idx)]
    for d in (0, 8):
        for sc in ("hi-Latn", "gold"):
            rows = data.correction(src, sc, d)
            enc = M.encode(tok, rows)
            for batch in run(enc):
                for i, pad, j in batch:
                    a, b = M.final_turn_start(tok, enc[i]), max(t for t, x in enumerate(enc[i]) if x == eot)
                    k, h = store["k"][j, pad + a:pad + b + 1].float(), store["h"][j, pad + a:pad + b + 1].float()
                    v = vec.setdefault((rows[i]["item_id"], d), {})
                    if sc == "hi-Latn":
                        A += k.double().T @ k.double() / k.shape[0]
                        v.update(kbar=k.mean(0), k_last=store["k"][j, -1].float(), xbar=h.mean(0), x_last=store["h"][j, -1].float())
                    else:
                        v.update(ybar=h.mean(0), y_last=store["h"][j, -1].float())
    rng = random.Random(0)
    ans = {x["item_id"]: x["response"] for x in data.jl(cfg["deva_answers"]) if x["switch"] == "hi-Deva" and x.get("single_turn")}
    seqs = [M.encode(tok, [x])[0] for x in data.correction(src, "en", 0)]
    seqs += [tok(tok.apply_chat_template(M.messages(x) + [{"role": "assistant", "content": ans[x["item_id"]][:1500]}], tokenize=False),
                 add_special_tokens=False)["input_ids"] for x in data.correction(src, "hi-Deva", 0)]
    seqs += [M.encode(tok, [x])[0] for x in data.correction(src, "hi-Deva", 8)]
    for batch in run(seqs):
        for i, pad, j in batch:
            pos = rng.sample(range(len(seqs[i])), min(24, len(seqs[i])))
            k = store["k"][j, [pad + p for p in pos]].double()
            C += k.T @ k
    for h in hk:
        h.remove()
    st = lambda f: torch.stack([v[f] for v in vec.values()]).double()  # noqa: E731
    kb, kl = st("kbar"), st("k_last")
    B = (st("ybar") - st("xbar")).T @ kb + (st("y_last") - st("x_last")).T @ kl
    A = A + kl.T @ kl
    Mx = A + cfg["mu"] * C
    Mx.diagonal().add_(1e-2 * torch.trace(A) / D)
    delta = torch.linalg.solve(Mx, B.T).T.float().cpu()
    torch.save({"delta": delta, "layer": L, "gamma": cfg["gamma"], "mu": cfg["mu"]}, cfg["out"])


# ------------------------------------------------------------------ export (hf_models)
HOOK = '''"""N+{arm} for Aya Expanse 8B (as evaluated): romanized Hindi -> Devanagari (normalize.py), then at layer 19 (output of
layers[18]) fire when w . h_last > c and add beta * g_bar * r_hat to every token of the final user turn (never to generated
tokens). Scope: inputs already labelled as romanized Hindi (no language-ID routing)."""
import json
import torch

def load(folder, depth="8"):
    a, cfg = torch.load(f"{{folder}}/steering.pt"), json.load(open(f"{{folder}}/steering.json"))
    return {{"w": a["w"], "r": a["r_hat"], "c": cfg["threshold_c"][depth], "push": cfg["beta"] * cfg["g_bar"], "layer": cfg["layer"]}}

@torch.no_grad()
def generate(model, tok, messages, s, **kw):
    """One conversation; the last message is the (normalized) user request."""
    ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
    t0, t1 = tok.convert_tokens_to_ids("<|START_OF_TURN_TOKEN|>"), tok.convert_tokens_to_ids("<|USER_TOKEN|>")
    start = max(i for i in range(ids.shape[1] - 1) if ids[0, i] == t0 and ids[0, i + 1] == t1)
    w, r = s["w"].to(model.device), s["r"].to(model.device)
    def hook(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        if h.shape[1] == 1:
            return out
        if float(h[0, -1].float() @ w) > s["c"]:
            h = h.clone(); h[:, start:] += (s["push"] * r).to(h.dtype)
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
    handle = model.model.layers[s["layer"] - 1].register_forward_hook(hook)
    try:
        out = model.generate(ids, do_sample=False, **kw)
    finally:
        handle.remove()
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
'''
NORM = '''"""IndicXlit normalization (own env: ai4bharat-transliteration, fairseq 0.12.2, Python 3.10)."""
from ai4bharat.transliteration import XlitEngine
_E = XlitEngine("hi", beam_width=4, rescore=True, src_script_type="en")

def normalize(text: str) -> str:
    return _E.translit_sentence(text)["hi"]
'''


def export(cfg):
    """Builds hf_models/<name> for N+Sw (vectors, thresholds, hook) and N+Rw (merged bf16 safetensors + fp32 delta + hook)."""
    import shutil
    from safetensors.torch import save_file
    th = json.loads((Path(cfg["fit"]) / "thresholds.json").read_text())
    w = torch.load(Path(cfg["fit"]) / "directions.pt")
    r = r_hat(cfg["r_hat"])
    for arm, weights in (("Sw", "base"), ("Rw", "read")):
        out = Path(cfg["out"]) / f"aya8b-hindi-safety-N-{arm}"
        out.mkdir(parents=True, exist_ok=True)
        t = th[weights]
        torch.save({"w": w[f"w_{weights}"].float(), "r_hat": r}, out / "steering.pt")
        (out / "steering.json").write_text(json.dumps({"layer": M.LAYER, "beta": cfg["beta"], "g_bar": t["g_bar"]["hi-Latn"]["g_bar"],
                                                       "threshold_c": {str(d): t["thresholds"][f"hi-Latn|{d}"]["w"]["c"] for d in (0, 8)},
                                                       "alpha": 0.05, "base_model": M.MODEL, "base_revision": M.REVISION}, indent=1))
        (out / "steer.py").write_text(HOOK.format(arm=arm))
        (out / "normalize.py").write_text(NORM)
        shutil.copyfile(Path(cfg["cards"]) / f"N-{arm}.md", out / "README.md")
        if arm == "Rw":
            d = torch.load(cfg["read_delta"], weights_only=False)
            save_file({"model.layers.16.mlp.down_proj.weight_delta": (d["gamma"] * d["delta"]).contiguous()}, out / "delta_W_down_L16_fp32.safetensors")
            if cfg.get("merge", True):
                merge(cfg["read_delta"], out / "merged")


def merge(delta_path, out: Path):
    """Base weights + gamma * delta on layers[16].mlp.down_proj (fp32 add, cast to bf16, as evaluated) -> safetensors."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(M.MODEL, revision=M.REVISION, dtype=torch.bfloat16, device_map="cpu")
    M.WeightEdit(model, delta_path).apply(1.0)
    model.save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(M.MODEL, revision=M.REVISION).save_pretrained(out)

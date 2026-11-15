"""EXPLORATORY (M1 was NO-GO) M2: detect with w, act with r_hat. HF, greedy, seed 42, batch 16, fixed full batches.

Conditions (83 correction sweep pairs, 166 rows per script). All runs of one depth run in ONE process:
unedited weights first, then the Read weight edit.
  Zb  base, no gate
  Sr  base + span gate on r_hat   (thresholds step3/gate_conformal_d{d}.json; = pilot c4 span alone)
  Sw  base + span gate on OOF w_base
  Zr  Read, no gate
  Rr  Read + span gate on r_hat   (thresholds read_final/gate_conformal_read_d{d}.json; = pilot c2)
  Rw  Read + span gate on OOF w_read
Push (every arm): h' = h + beta g r_hat on the final user turn (START_OF_TURN .. last prompt token), beta 16, no
generated token. Fire = 1[s_det > c_det]. r_hat arms: g = max(0, r_hat.h - c_r) (legacy gate_hook, unchanged).
w arms: g = k max(0, w_f.h - c_w), k = sd_r / sd_w (benign calibration sds, explore_prep.py), w_f = the M1 fold
direction (b, lambda 0) that did not train on the row's pair (NOTES.md, design decision). c_w from the pooled OOF
benign scores of the 250 non-sweep pairs (explore_prep.py thresholds.json).
Checks (checks-<tag>.json):
  folds  the refit fold directions reproduce M1's stored OOF scores (max |diff| < 1e-8);
  path   the per-row hook with r_hat in every fold slot and k = 1 reproduces the first 12 replies of the first r_hat
         arm per model in the same batch (exact). The run stops after generation if it fails.
  stored (optional, --stored) Sr / Rr replies vs pilot c4 / c2 files of the same depth, by conversation key.
Level 1 (unchanged): benign Zb replies teacher-forced under base and Read weights, no gate.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "refusal_direction"))

import m1  # noqa: E402
from cache import DEPTHS as CACHE_DEPTHS, LAYERS, SCRIPTS as CACHE_SCRIPTS, environment  # noqa: E402
from calib_depth import scores as layer_scores  # noqa: E402
from extract import encode, load_model  # noqa: E402
from gen import get_rows  # noqa: E402
from intervene import final_message, gate_hook, generate  # noqa: E402
from ov_common import SCORE_LAYER, XLIT, folds, r_hat  # noqa: E402
from prompts import ROOT  # noqa: E402
from read_pilot import BETA, READ, SEED, WeightEdit, forced, level1, reply_ids, save  # noqa: E402

LABEL = "EXPLORATORY (M1 was NO-GO)"
UNEDITED = ("Zb", "Sr", "Sw", "Sm")
EDITED = ("Zr", "Rr", "Rw", "Rm")  # Sm / Rm: matched-push follow-up (w gate fires, push = beta g_match r_hat)
MATCH_REF = {"Sm": "Sr", "Rm": "Rr"}
MATCH_SCRIPT = {"hi-Deva": "hi-Latn"}  # user, 2026-11-11: no r_hat arm in hi-Deva; match the hi-Latn push
MODEL_OF = {c: "base" if c in UNEDITED else "read" for c in UNEDITED + EDITED}
R_FILES = {"base": ROOT / "runs/overnight_v2/step3/gate_conformal_d{d}.json",
           "read": READ / "gate_conformal_read_d{d}.json"}
PILOT_NAME = {"Sr": "c4", "Rr": "c2"}


def fold_directions(cache, name, pairs, df):
    """The 5 M1 fold directions (b, lambda 0) of one model, checked against M1's stored OOF scores."""
    fold_of = folds(pairs)
    fi = np.array([fold_of[p] for p in pairs])
    X = torch.load(cache / name / "h.pt", mmap=True)[..., LAYERS.index(SCORE_LAYER), :].double()
    W = torch.stack([m1.solve(m1.matrices(X, np.where(fi != f)[0]), "b", 0.0)[0] for f in range(5)])
    mine = torch.einsum("pasdf,pf->pasd", X, W[fi])
    sel = df[(df.model == name) & (df.version == "b") & (df["lambda"] == 0.0) & ~df.control]
    pos = {p: i for i, p in enumerate(pairs)}
    ref = torch.full_like(mine, float("nan"))
    for p, a, s, d, v in zip(sel.pair_id, sel.harmful, sel.script, sel.depth, sel.s_inv):
        ref[pos[p], a, CACHE_SCRIPTS.index(s), CACHE_DEPTHS.index(d)] = v
    err = float((mine - ref).abs().max())
    return W, fold_of, {"max_abs_diff_vs_m1_oof": err, "tol": 1e-8, "pass": err < 1e-8,
                        "pairwise_cos": (W @ W.T).tolist()}


def oof_gate_hook(model, R, r_add, beta, state, g_push=None):
    """Span gate with a per-row detection direction R[state['fold'][row]] (R: [5, F], scale k included).
    Mirrors gate_hook(span=True): every R[f] is applied to the whole batch with the same matrix-vector product as
    the legacy gate, then each row takes its own fold's score."""
    def hook(_module, _inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        if h.shape[1] == 1 and "g" in state:  # decoding: generated tokens are never edited
            return output
        s = h.shape[1] - 1
        scores = torch.stack([h[:, s].float() @ R[f] for f in range(R.shape[0])])     # [5, b]
        score = scores.gather(0, state["fold"][None].to(scores.device))[0]
        state["g"] = (score - state["c"]).clamp(min=0)
        idx = torch.arange(h.shape[1], device=h.device)
        mask = (idx[None, :] >= state["span_start"].to(h.device)[:, None]) & (idx[None, :] <= s)
        push = state["g"] if g_push is None else g_push * (state["g"] > 0).float()  # g_push: matched push size
        delta = (beta * push)[:, None, None] * mask[..., None].float() * r_add
        h = h + delta.to(h.dtype)
        return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h
    return [model.model.layers[SCORE_LAYER - 1].register_forward_hook(hook)]


def parse_runs(items, depth):
    runs = []
    for it in items:
        script, cond = it.split(":")
        assert cond in MODEL_OF, cond
        assert script in CACHE_SCRIPTS, script
        assert not (cond in ("Sr", "Rr") and script not in ("hi-Latn", "en-hi")), "r_hat thresholds: hi-Latn, en-hi"
        assert not (cond in MATCH_REF and script not in ("hi-Latn", "en-hi", "hi-Deva")), "matched push: hi-Latn, en-hi, hi-Deva"
        assert not (cond in MATCH_REF and depth != 8), "matched push: depth 8 only"
        runs.append((script, cond))
    return [x for x in runs if x[1] in UNEDITED] + [x for x in runs if x[1] in EDITED]


def first_batch(tok, rows, order, batch_size):
    if order is None:  # the default order of generate()
        encoded = encode(tok, rows)
        order = sorted(range(len(rows)), key=lambda i: -len(encoded[i][0]))
    return order[:batch_size]


def path_check(tok, model, rows, results, order, r, args):
    """The new per-row hook, given r_hat in every slot (k = 1), must reproduce the legacy r_hat gate exactly."""
    first = first_batch(tok, rows, order, args.batch_size)
    sub = [dict(rows[i], gate_fold=j % 5) for j, i in enumerate(first)]  # every fold slot used, all equal r_hat
    state = {}
    handles = oof_gate_hook(model, r[None].expand(5, -1).contiguous(), r, BETA, state)
    try:
        again = generate(tok, model, sub, args.batch_size, args.max_new_tokens, state, list(range(len(sub))))
    finally:
        for h in handles:
            h.remove()
    same = [again[j][:2] == results[i][:2] for j, i in enumerate(first)]
    n = min(12, len(same))
    return {"n_batch": len(same), "identical_first_12": sum(same[:n]), "pass": all(same[:n])}


def stored_check(path, rows, results):
    if not path.exists():
        return {"file": str(path), "found": False}
    ref = {}
    for line in path.read_text().splitlines():
        x = json.loads(line)
        ref[x["conversation_id"]] = x["response"]
    same = [ref.get(row["key"]) == res[0] for row, res in zip(rows, results) if row["key"] in ref]
    return {"file": str(path), "found": True, "n_compared": len(same), "identical": sum(same)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True, help="writes <out>/d<depth>/<script>-<cond>.jsonl")
    p.add_argument("--depth", type=int, choices=[0, 5, 8], required=True)
    p.add_argument("--runs", nargs="+", required=True, help="script:cond, e.g. hi-Latn:Sw")
    p.add_argument("--tag", required=True)
    p.add_argument("--prep", type=Path, default=ROOT / "runs/multiturn_v2/m2_explore/prep")
    p.add_argument("--m1", type=Path, default=ROOT / "runs/multiturn_v2/m1")
    p.add_argument("--cache", type=Path, default=ROOT / "runs/multiturn_v2/m0_cache")
    p.add_argument("--stored", type=Path, help="optional pilot read_pilot.py output folder (d<depth>/<script>-c2/c4)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--limit-pairs", type=int, help="smoke tests only")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    runs = parse_runs(args.runs, args.depth)
    folder = args.out / f"d{args.depth}"
    for script, cond in runs:
        assert not (folder / f"{script}-{cond}.jsonl").exists(), f"output exists: {script}-{cond}"
        assert cond not in MATCH_REF or (folder / f"{MATCH_SCRIPT.get(script, script)}-{MATCH_REF[cond]}.jsonl").exists(), f"{cond} needs {MATCH_REF.get(cond)}"
        get_rows(script, args.depth, "correction", "v2", args.limit_pairs)
    prep_checks = json.loads((args.prep / "checks.json").read_text())
    assert prep_checks["base:3.2_conformal"]["pass"] and prep_checks["read:3.2_conformal"]["pass"]
    thr_w = json.loads((args.prep / "thresholds.json").read_text())
    thr_r = {m: json.loads(Path(str(f).format(d=args.depth)).read_text())["c_r"] for m, f in R_FILES.items()}
    print(f"dry run ok: depth {args.depth}, {len(runs)} runs", flush=True)
    if args.dry_run:
        return

    from setproctitle import setproctitle
    setproctitle(f"aya_alignment:m2x_{args.tag}")
    folder.mkdir(parents=True, exist_ok=True)
    pairs = json.loads((args.cache / "pairs.json").read_text())
    df = pd.read_parquet(args.m1 / "oof_scores.parquet")
    checks, W, fold_of = {}, {}, None
    for name in sorted({MODEL_OF[c] for _, c in runs if c in ("Sw", "Rw", "Sm", "Rm")}):
        W[name], fold_of, checks[f"folds_{name}"] = fold_directions(args.cache, name, pairs, df)
    if not all(v["pass"] for v in checks.values()):
        (folder / f"checks-{args.tag}.json").write_text(json.dumps(checks, indent=1))
        raise SystemExit(f"VERIFY failed before generation: {checks}")

    tok, model = load_model()
    torch.manual_seed(SEED)
    edit = WeightEdit(model, READ / "delta_final.pt")
    r = r_hat(model.device)
    meta_common = {"label": LABEL, "stage": "m2_explore", "split": "correction sweep pairs (83)", "pool": "v2",
                   "depth": args.depth, "beta": BETA, "push": "beta g r_hat, final user turn span, no generated token",
                   "thresholds_r_hat": {m: {"file": str(f).format(d=args.depth), "c": thr_r[m]} for m, f in R_FILES.items()},
                   "thresholds_w": str(args.prep / "thresholds.json"),
                   "detector_w": "OOF per pair: M1 fold direction (b, lambda 0) not trained on the pair",
                   "batch_size": args.batch_size, "max_new_tokens": args.max_new_tokens, "seed": SEED}
    with (folder / f"env-{args.tag}.json").open("x") as f:
        json.dump(environment(args.out) | meta_common | {"runs": args.runs, "tag": args.tag}, f, indent=1, default=str)

    ref, lvl = {}, {}
    for script, cond in runs:
        name = MODEL_OF[cond]
        edited = name == "read"
        if edited and not edit.on:
            edit.apply(1.0)
        assert edit.on == edited
        rows = get_rows(script, args.depth, "correction", "v2", args.limit_pairs)
        order = final_message(tok, rows, "romanized", XLIT) if script == "hi-Latn" else None
        for row, v in zip(rows, layer_scores(tok, model, rows, r, SCORE_LAYER, args.batch_size)):
            row["l19_score"] = v
        print(f"start {script} {cond} depth {args.depth} ({len(rows)} rows)", flush=True)

        state, handles, gate = {}, [], None
        if cond in ("Sr", "Rr"):
            for row in rows:
                row["gate_c"] = thr_r[name][script]
            handles = gate_hook(model, SCORE_LAYER, r, r, 0.0, BETA, state, None, True)
            gate = {"detector": "r_hat", "c": thr_r[name][script], "k": 1.0}
        elif cond in ("Sw", "Rw", "Sm", "Rm"):
            t = thr_w[name][f"{script}|{args.depth}"]
            for row in rows:
                row["gate_c"] = t["k_gate"] * t["c"]
                row["gate_fold"] = fold_of[row["pair_id"]]
            R = (t["k_gate"] * W[name]).float().to(model.device)
            g_match = None
            if cond in MATCH_REF:  # mean gate_g over the fired rows of the r_hat arm, measured from its generated file
                ref_file = folder / f"{MATCH_SCRIPT.get(script, script)}-{MATCH_REF[cond]}.jsonl"
                fired = [x["gate_g"] for x in map(json.loads, ref_file.read_text().splitlines()) if x["gate_g"] > 0]
                g_match = sum(fired) / len(fired)
            handles = oof_gate_hook(model, R, r, BETA, state, g_match)
            gate = {"detector": f"OOF w_{name}", "c_w": t["c"], "k": t["k_gate"], "sd_r": t["sd_r"], "sd_w": t["sd_w"]}
            if g_match is not None:
                gate |= {"push": "matched", "g_match": g_match, "g_match_from": str(ref_file), "n_fired_ref": len(fired)}
        try:
            results = generate(tok, model, rows, args.batch_size, args.max_new_tokens, state if handles else None, order)
        finally:
            for h in handles:
                h.remove()

        meta = {**meta_common, "cond": cond, "model": name, "weights_edited": edited, "edit_layer": 16, "gate": gate,
                "final_message": "romanized" if script == "hi-Latn" else None,
                "batch_order": "fixed: decreasing romanized prompt length" if order is not None else "decreasing prompt length"}
        save(folder / f"{script}-{cond}.jsonl", rows, results, cond, meta, args.max_new_tokens)

        if cond in ("Sr", "Rr") and f"path_{name}" not in checks:
            checks[f"path_{name}"] = path_check(tok, model, rows, results, order, r, args)
            print("path check", name, checks[f"path_{name}"], flush=True)
        if cond in ("Sr", "Rr") and args.stored:
            checks[f"stored_{script}_{cond}"] = stored_check(
                args.stored / f"d{args.depth}" / f"{script}-{PILOT_NAME[cond]}.jsonl", rows, results)
        ben = [i for i, row in enumerate(rows) if row["request_type"] == "benign"]
        if cond == "Zb":
            reps = reply_ids(tok, [results[i] for i in ben])
            ref[script] = ([rows[i] for i in ben], reps, forced(tok, model, [rows[i] for i in ben], reps))
        if edited and script in ref and script not in lvl:
            brow, reps, base = ref[script]
            lvl[script] = level1(base, forced(tok, model, brow, reps))

    if lvl:
        with (folder / f"level1-{args.tag}.json").open("x") as f:
            json.dump({"depth": args.depth, "definition": "benign prompts; Zb replies teacher-forced; KL(P_Zb || P_Read) "
                       "of the first token and ppl ratio, Read weights without gates", **lvl}, f, indent=1)
    if edit.on:
        edit.revert()
    (folder / f"checks-{args.tag}.json").write_text(json.dumps(checks, indent=1, default=str))
    failed = [k for k, v in checks.items() if "pass" in v and not v["pass"]]
    if failed:
        raise SystemExit(f"VERIFY failed: {failed}")
    print("done", folder, flush=True)


if __name__ == "__main__":
    main()

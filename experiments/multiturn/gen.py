"""Stages M2 and M3: generation with the context-invariant gate. HF, greedy, seed 42, batch 16, fixed full batches.

All conditions of one setting (stage, split, pool, depth) run in ONE process: unedited runs first, then Read weights.
  Z  zero edit (unedited weights, no hook)
  N  normalization: IndicXlit final message, unedited (hi-Latn only; M3 reference)
  A  Read + span gate on s_ctx = r_hat . h, beta 16, thresholds read_final/gate_conformal_read_d{gd}.json (current method)
  B  Read + span gate on s_inv = w . h, beta 16; the edit adds beta * g * r_hat to the final user turn
  C  Read + gate on s_inv + minimal-norm push: delta* = 1[s_inv > c] max(0, s* - r_c . h) r_c on every final-turn
     position; r_c = mean-removed r_hat (unit); s* = 25th percentile of English harmful r_c . h at depth gd
     (Read model, 333 correction pairs, M0 cache)
  U  upper reference (M2, depth 8): Read + span gate on s_iso, the r_hat score of the same request in single-turn form
     (a second forward pass). Implemented with gate_hook, a zero gate vector and the per-row threshold c - s_iso,
     which gives g = max(0, s_iso - c) exactly. c = Read depth-0 threshold (gate_conformal_read_d0.json).
Thresholds for s_inv: c = s_(k), k = ceil((n + 1)(1 - alpha)), alpha 0.05, benign OOF s_inv scores of the 250 non-sweep
correction pairs at (script, gd) (M1 oof_scores.parquet, selected version and lambda). The gate itself uses w_final.
Checks (checks-<tag>.json; the run stops before generation if M2a or M2d fails, and after it if M2b or M2c fails):
  M2a LayerNorm invariance and mean(r_c) = 0; M2b minimal-norm push; M2c zero-edit reproduction; M2d conformal sanity.
Level 1: benign Z replies teacher-forced under unedited and Read weights (the weight edit only; gates are not active).
"""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "refusal_direction"))

from cache import DEPTHS as CACHE_DEPTHS, LAYERS, SCRIPTS as CACHE_SCRIPTS, environment  # noqa: E402
from calib_depth import scores as layer_scores  # noqa: E402
from extract import encode, load_model  # noqa: E402
from intervene import final_message, gate_hook, generate  # noqa: E402
from ov_common import SCORE_LAYER, XLIT, r_hat, rows_for, sweep_pairs  # noqa: E402
from prompts import ROOT  # noqa: E402
from read_pilot import BETA, READ, SEED, WeightEdit, forced, level1, reply_ids, save  # noqa: E402

ALPHA = 0.05
UNEDITED = ("Z", "N")
EDITED = ("A", "B", "C", "U")
TOL_PUSH, TOL_NORM = 1e-3, 1e-6


def get_rows(script, depth, split, pool, limit=None):
    """M2: the 83 correction sweep pairs; M3: all 83 held-out pairs. 166 rows per script."""
    sweep = sweep_pairs() if split == "correction" else None
    rows = [r for r in rows_for(script, depth, split, pool) if sweep is None or r["pair_id"] in sweep]
    if limit:
        keep = sorted({r["pair_id"] for r in rows})[:limit]
        rows = [r for r in rows if r["pair_id"] in keep]
    else:
        assert len(rows) == 166, (script, depth, len(rows))
    return rows


def conformal(scores, alpha):
    """Split-conformal threshold. If k > n the threshold is +inf (too few calibration scores for this alpha)."""
    n = len(scores)
    k = math.ceil((n + 1) * (1 - alpha))
    c = sorted(scores)[k - 1] if k <= n else float("inf")
    fire = sum(s > c for s in scores) / n
    return {"c": c, "n": n, "k": k, "alpha": alpha, "calibration_fire_rate": fire, "bound": alpha + 1 / (n + 1),
            "pass": fire <= alpha + 1 / (n + 1)}


def thresholds(m1_dir, gd, scripts, alpha):
    """s_inv thresholds per script at depth gd from the M1 out-of-fold benign scores of the non-sweep pairs."""
    go = json.loads((m1_dir / "go.json").read_text())
    df = pd.read_parquet(m1_dir / "oof_scores.parquet")
    sel = df[(df.model == "read") & (df.version == go["best"]["version"]) & (df["lambda"] == go["best"]["lambda"])
             & ~df.control & (df.harmful == 0) & (df.depth == gd) & ~df.pair_id.isin(sweep_pairs())]
    return {s: conformal(sel[sel.script == s].s_inv.tolist(), alpha) for s in scripts}


def centred_direction(r):
    rc = r.double() - r.double().mean()
    return rc / rc.norm()


def s_star(cache, r_c, gd):
    """25th percentile of English harmful r_c . h (Read model, layer 19, depth gd, all correction pairs)."""
    h = torch.load(cache / "read" / "h.pt", mmap=True)
    x = h[:, 1, CACHE_SCRIPTS.index("en"), CACHE_DEPTHS.index(gd), LAYERS.index(SCORE_LAYER)].double()
    return float(torch.quantile(x @ r_c, 0.25))


def push_hook(model, w, r_c, s_target, state):
    """Condition C, at the output of layers[18] (= hidden_states[19]). Decides at the last prompt token in the
    prefill call and adds the same delta* to every final-turn position. Generated tokens are not edited."""
    w32, rc32 = w.float(), r_c.float()

    def hook(_module, _inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        if h.shape[1] == 1 and "g" in state:
            return output
        s = h.shape[1] - 1
        hs = h[:, s].float()
        fired = (hs @ w32) > state["c"]
        push = (s_target - hs @ rc32).clamp(min=0) * fired
        delta = push[:, None] * rc32
        state["g"] = push
        state.setdefault("m2b", []).append({
            "fired": fired.tolist(), "after": ((hs + delta) @ rc32).tolist(), "push": push.tolist(),
            "delta_norm": delta.norm(dim=1).tolist(),
            "after_bf16": ((h[:, s] + delta.to(h.dtype)).float() @ rc32).tolist()})
        idx = torch.arange(h.shape[1], device=h.device)
        mask = (idx[None] >= state["span_start"].to(h.device)[:, None]) & (idx[None] <= s)
        h = h + (mask[..., None].float() * delta[:, None]).to(h.dtype)
        return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h

    return [model.model.layers[SCORE_LAYER - 1].register_forward_hook(hook)]


def m2a(model, r_c):
    """LayerNorm invariance to adding t * 1 (t = 5) for 10 random h at the input norm of layers[19]; mean(r_c) = 0."""
    ln = model.model.layers[SCORE_LAYER].input_layernorm
    gen = torch.Generator().manual_seed(0)
    h = (torch.randn(10, r_c.shape[0], generator=gen) * 10).to(model.device)
    fp32 = float((ln(h + 5) - ln(h)).norm(dim=1).max())
    hb = h.to(torch.bfloat16)
    bf16 = float((ln(hb + 5).float() - ln(hb).float()).norm(dim=1).max())
    mean = abs(float(r_c.mean()))
    return {"ln_shift_max_fp32": fp32, "ln_shift_max_bf16_info": bf16, "tol": 1e-3, "mean_r_c": mean,
            "pass": fp32 < 1e-3 and mean < 1e-10,
            "note": "computed in fp32: a bf16 input is rounded after adding t, which alone moves LN by more than 1e-3"}


def m2b(records):
    fired = [(a, p, n) for rec in records for f, a, p, n in zip(rec["fired"], rec["after"], rec["push"], rec["delta_norm"]) if f]
    return fired


def parse_runs(items, stage, depth, winner):
    allowed = {"m2": {"Z", "A", "B", "C", "U"}, "m3": {"Z", "A", "N"} | ({winner} if winner else set())}[stage]
    runs = []
    for it in items:
        script, cond = it.split(":")
        assert cond in allowed, f"{cond} not allowed in {stage}"
        assert not (cond == "A" and script not in ("hi-Latn", "en-hi")), "A thresholds exist for hi-Latn and en-hi only"
        assert not (cond == "N" and script != "hi-Latn"), "normalization is hi-Latn only"
        assert not (cond == "U" and depth != 8), "upper reference is depth 8 only"
        runs.append((script, cond))
    return [x for x in runs if x[1] in UNEDITED] + [x for x in runs if x[1] in EDITED]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=["m2", "m3"], required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--depth", type=int, choices=[0, 2, 5, 8], required=True)
    p.add_argument("--pool", choices=["v1", "v2"], default="v2")
    p.add_argument("--gate-depth", type=int, help="depth of thresholds and s* (default --depth)")
    p.add_argument("--runs", nargs="+", required=True, help="script:cond, e.g. hi-Latn:C")
    p.add_argument("--winner", choices=["B", "C"], help="M3: the M2 winner")
    p.add_argument("--m1", type=Path, default=ROOT / "runs/multiturn_v2/m1")
    p.add_argument("--cache", type=Path, default=ROOT / "runs/multiturn_v2/m0_cache")
    p.add_argument("--tag", required=True)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--limit-pairs", type=int, help="smoke tests only")
    p.add_argument("--alpha", type=float, default=ALPHA, help="s_inv conformal level; only smoke tests change it")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    split = "correction" if args.stage == "m2" else "heldout"
    gd = args.gate_depth if args.gate_depth is not None else args.depth
    runs = parse_runs(args.runs, args.stage, args.depth, args.winner)
    folder = args.out / f"d{args.depth}"
    for script, cond in runs:
        assert not (folder / f"{script}-{cond}.jsonl").exists(), f"output exists: {script}-{cond}"
        get_rows(script, args.depth, split, args.pool, args.limit_pairs)
    for f in (args.m1 / "go.json", args.m1 / "w_final.pt", args.cache / "read" / "h.pt", READ / "delta_final.pt"):
        assert f.exists(), f"missing input {f}"
    go = json.loads((args.m1 / "go.json").read_text())
    if not args.limit_pairs:
        assert go["GO"], "M1 is not GO: M2/M3 must not run"
        assert args.alpha == ALPHA, "alpha 0.05 is fixed for real runs"
    scripts = sorted({s for s, _ in runs})
    thr = thresholds(args.m1, gd, scripts, args.alpha)
    print(f"dry run ok: stage {args.stage}, {split}, pool {args.pool}, depth {args.depth} (gate depth {gd}), "
          f"{len(runs)} runs", flush=True)
    if args.dry_run:
        return

    from setproctitle import setproctitle
    setproctitle(f"aya_alignment:{args.stage}_{args.tag}")
    folder.mkdir(parents=True, exist_ok=True)
    tok, model = load_model()
    torch.manual_seed(SEED)
    edit = WeightEdit(model, READ / "delta_final.pt")
    r = r_hat(model.device)
    r_c = centred_direction(r.cpu())
    wf = torch.load(args.m1 / "w_final.pt", weights_only=False)
    w = wf["w"].float().to(model.device)
    star = s_star(args.cache, r_c, gd)
    gate_A = json.loads((READ / f"gate_conformal_read_d{gd}.json").read_text())
    gate_A0 = json.loads((READ / "gate_conformal_read_d0.json").read_text())

    checks = {"M2a": m2a(model, r_c.to(model.device).float()),
              "M2d": {s: v for s, v in thr.items()}}
    checks["M2d_pass"] = all(v["pass"] for v in thr.values())
    meta_common = {"stage": args.stage, "split": split, "pool": args.pool, "depth": args.depth, "gate_depth": gd,
                   "w_final": {k: v for k, v in wf.items() if k != "w"}, "m1_best": go["best"], "s_star": star,
                   "thresholds_s_inv": {s: v["c"] for s, v in thr.items()},
                   "thresholds_A": gate_A["c_r"], "threshold_U_depth0": gate_A0["c_r"], "beta": BETA,
                   "batch_size": args.batch_size, "max_new_tokens": args.max_new_tokens, "seed": SEED}
    env = environment(args.out) | meta_common | {"runs": args.runs, "tag": args.tag}
    with (folder / f"env-{args.tag}.json").open("x") as f:
        json.dump(env, f, indent=1, default=str)
    if not (checks["M2a"]["pass"] and checks["M2d_pass"]):
        (folder / f"checks-{args.tag}.json").write_text(json.dumps(checks, indent=1))
        raise SystemExit(f"VERIFY failed before generation: {checks}")

    ref, lvl, m2b_records = {}, {}, []
    for script, cond in runs:
        edited = cond in EDITED
        if edited and not edit.on:
            edit.apply(1.0)
        assert edit.on == edited
        rows = get_rows(script, args.depth, split, args.pool, args.limit_pairs)
        kind = "auto" if cond == "N" else "romanized"
        order = final_message(tok, rows, kind, XLIT) if script == "hi-Latn" else None
        for row, v in zip(rows, layer_scores(tok, model, rows, r, SCORE_LAYER, args.batch_size)):
            row["l19_score"] = v
        for row, v in zip(rows, layer_scores(tok, model, rows, w, SCORE_LAYER, args.batch_size)):
            row["s_inv"] = v
        print(f"start {script} {cond} depth {args.depth} ({len(rows)} rows)", flush=True)

        state, handles = {}, []
        if cond == "A":
            for row in rows:
                row["gate_c"] = gate_A["c_r"][script]
            handles = gate_hook(model, SCORE_LAYER, r, r, 0.0, BETA, state, None, True)
        elif cond == "B":
            for row in rows:
                row["gate_c"] = thr[script]["c"]
            handles = gate_hook(model, SCORE_LAYER, w, r, 0.0, BETA, state, None, True)
        elif cond == "C":
            for row in rows:
                row["gate_c"] = thr[script]["c"]
            handles = push_hook(model, w, r_c.to(model.device), star, state)
        elif cond == "U":
            single = {x["item_id"]: x for x in rows_for(script, 0, split)}
            iso_rows = [dict(single[row["item_id"]]) for row in rows]
            for row, v in zip(rows, layer_scores(tok, model, iso_rows, r, SCORE_LAYER, args.batch_size)):
                row["s_iso"] = v
                row["gate_c"] = gate_A0["c_r"][script] - v  # g = max(0, 0 . h - (c - s_iso)) = max(0, s_iso - c)
            handles = gate_hook(model, SCORE_LAYER, torch.zeros_like(r), r, 0.0, BETA, state, None, True)
        try:
            results = generate(tok, model, rows, args.batch_size, args.max_new_tokens, state if handles else None, order)
        finally:
            for h in handles:
                h.remove()
        if cond == "C":
            m2b_records += state.get("m2b", [])

        meta = {**meta_common, "cond": cond, "weights_edited": edited, "edit_layer": 16,
                "gate": {"A": "s_ctx", "B": "s_inv", "C": "s_inv + minimal-norm push", "U": "s_iso"}.get(cond),
                "final_message": kind if script == "hi-Latn" else None,
                "batch_order": "fixed: decreasing romanized prompt length" if order is not None else "decreasing prompt length"}
        save(folder / f"{script}-{cond}.jsonl", rows, results, cond, meta, args.max_new_tokens)

        ben = [i for i, row in enumerate(rows) if row["request_type"] == "benign"]
        if cond == "Z":
            reps = reply_ids(tok, [results[i] for i in ben])
            ref[script] = ([rows[i] for i in ben], reps, forced(tok, model, [rows[i] for i in ben], reps))
            if "M2c" not in checks:
                checks["M2c"] = zero_edit_check(tok, model, edit, rows, results, order, w, r_c, star, args)
        if edited and script in ref and script not in lvl:
            brow, reps, base = ref[script]
            lvl[script] = level1(base, forced(tok, model, brow, reps))

    if m2b_records:
        fired = m2b(m2b_records)
        gap = min((a - star for a, _, _ in fired), default=0.0)
        norm_err = max((abs(n - p) for _, p, n in fired), default=0.0)
        checks["M2b"] = {"n_fired": len(fired), "min_after_minus_s_star_fp32": gap, "tol": TOL_PUSH,
                         "max_norm_err": norm_err, "tol_norm": TOL_NORM,
                         "pass": gap >= -TOL_PUSH and norm_err <= TOL_NORM,
                         "min_after_minus_s_star_bf16_info": min((a - star for rec in m2b_records
                                                                  for f, a in zip(rec["fired"], rec["after_bf16"]) if f),
                                                                 default=0.0)}
    if lvl:
        with (folder / f"level1-{args.tag}.json").open("x") as f:
            json.dump({"depth": args.depth, "definition": "benign prompts; Z replies teacher-forced; KL(P_Z || P_Read) "
                       "of the first token and ppl ratio, Read weights without gates", **lvl}, f, indent=1)
    if edit.on:
        edit.revert()
    (folder / f"checks-{args.tag}.json").write_text(json.dumps(checks, indent=1, default=str))
    failed = [k for k in ("M2b", "M2c") if k in checks and not checks[k]["pass"]]
    if failed:
        raise SystemExit(f"VERIFY failed: {failed}")
    print("done", folder, flush=True)


def zero_edit_check(tok, model, edit, rows, results, order, w, r_c, star, args):
    """M2c / M3a: the first batch again, with the Read edit at scale 0 and the C hook installed but unable to fire.
    The first 12 replies must equal the Z replies (same prompts, same batch membership and order)."""
    if order is None:  # the default order of generate()
        encoded = encode(tok, rows)
        order = sorted(range(len(rows)), key=lambda i: -len(encoded[i][0]))
    first = order[:args.batch_size]
    sub = [dict(rows[i], gate_c=float("inf")) for i in first]
    edit.apply(0.0)
    state = {}
    handles = push_hook(model, w, r_c.to(model.device), star, state)
    try:
        again = generate(tok, model, sub, args.batch_size, args.max_new_tokens, state, list(range(len(sub))))
    finally:
        for h in handles:
            h.remove()
    edit.revert()
    same = [again[j][0] == results[i][0] and again[j][1] == results[i][1] for j, i in enumerate(first)]
    n = min(12, len(same))
    return {"n_batch": len(same), "identical_first_12": sum(same[:n]), "pass": all(same[:n])}


if __name__ == "__main__":
    main()

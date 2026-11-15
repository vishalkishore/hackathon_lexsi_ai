"""EXPLORATORY Step E (NOTES.md, STEP D): held-out run, w gate + fixed push vs r_hat gate. HF, greedy, seed 42,
batch 16, fixed full batches. All runs of one setting run in ONE process: unedited weights first, then Read weights.
  Zb  base, no gate                     Zr  Read, no gate
  Sr  base + r_hat span gate (beta 16)  Rr  Read + r_hat span gate (beta 16)
  Sw  base + w gate, fixed push         Rw  Read + w gate, fixed push
  N   normalization: IndicXlit final message, base, no gate (hi-Latn only)
w arms: fire = 1[w . h > c_w] (calib_w: w fit on the 250 non-sweep pairs, c_w per (script, gate depth) from the 83
sweep pairs, k = 80); push = beta * g_bar * r_hat on the final user turn span, g_bar one constant per (model, script)
= mean gate_g of the fired rows of the depth-8 r_hat arm on the sweep pairs (m2_explore/d8); en / hi-Deva use hi-Latn.
Implemented with explore_gen.oof_gate_hook (w in every slot, g_push = g_bar). Checks (checks-<tag>.json):
  path  the hook with r_hat in every slot and no fixed push reproduces the first 12 replies of the first r_hat arm.
Level 1 (unchanged): benign Zb replies teacher-forced under base and Read weights, no gate.
"""
import argparse
import json
from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "refusal_direction"))

from cache import environment  # noqa: E402
from calib_depth import scores as layer_scores  # noqa: E402
from explore_gen import R_FILES, oof_gate_hook, path_check  # noqa: E402
from extract import load_model  # noqa: E402
from gen import get_rows  # noqa: E402
from intervene import final_message, gate_hook, generate  # noqa: E402
from ov_common import SCORE_LAYER, XLIT, r_hat  # noqa: E402
from prompts import ROOT  # noqa: E402
from read_pilot import BETA, READ, SEED, WeightEdit, forced, level1, reply_ids, save  # noqa: E402

LABEL = "EXPLORATORY held-out (STEP D)"
UNEDITED, EDITED = ("Zb", "N", "Sr", "Sw"), ("Zr", "Rr", "Rw")
MODEL_OF = {c: "base" if c in UNEDITED else "read" for c in UNEDITED + EDITED}
GBAR_REF = {"base": "Sr", "read": "Rr"}
SETTINGS = {"v2d8": ("v2", 8, 8), "v2d0": ("v2", 0, 0), "v1d5": ("v1", 5, 5)}  # pool, depth, gate depth


def g_bar(explore, name, script):
    src = explore / "d8" / f"{'hi-Latn' if script in ('en', 'hi-Deva') else script}-{GBAR_REF[name]}.jsonl"
    fired = [x["gate_g"] for x in map(json.loads, src.read_text().splitlines()) if x["gate_g"] > 0]
    return sum(fired) / len(fired), str(src), len(fired)


def parse_runs(items):
    runs = []
    for it in items:
        script, cond = it.split(":")
        assert cond in MODEL_OF, cond
        assert script in ("hi-Latn", "en-hi", "en", "hi-Deva"), script
        assert not (cond in ("Sr", "Rr") and script not in ("hi-Latn", "en-hi")), "r_hat thresholds: hi-Latn, en-hi"
        assert not (cond == "N" and script != "hi-Latn"), "normalization is hi-Latn only"
        runs.append((script, cond))
    return [x for x in runs if x[1] in UNEDITED] + [x for x in runs if x[1] in EDITED]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True, help="writes <out>/<setting>/<script>-<cond>.jsonl")
    p.add_argument("--setting", choices=list(SETTINGS), required=True)
    p.add_argument("--runs", nargs="+", required=True, help="script:cond, e.g. hi-Latn:Sw")
    p.add_argument("--tag", required=True)
    p.add_argument("--calib", type=Path, default=ROOT / "runs/multiturn_v2/calib_w")
    p.add_argument("--explore", type=Path, default=ROOT / "runs/multiturn_v2/m2_explore")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--limit-pairs", type=int, help="smoke tests only")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    pool, depth, gd = SETTINGS[args.setting]
    runs = parse_runs(args.runs)
    folder = args.out / args.setting
    for script, cond in runs:
        assert not (folder / f"{script}-{cond}.jsonl").exists(), f"output exists: {script}-{cond}"
        get_rows(script, depth, "heldout", pool, args.limit_pairs)
    calib = json.loads((args.calib / "calib_w.json").read_text())
    assert calib["base"]["pass"] and calib["read"]["pass"], "Decision 1 checks failed"
    thr_w = json.loads((args.calib / "thresholds.json").read_text())
    thr_r = {m: json.loads(Path(str(f).format(d=gd)).read_text())["c_r"] for m, f in R_FILES.items()}
    gbars = {(MODEL_OF[c], s): g_bar(args.explore, MODEL_OF[c], s) for s, c in runs if c in ("Sw", "Rw")}
    print(f"dry run ok: {args.setting} (pool {pool}, depth {depth}, gate depth {gd}), {len(runs)} runs; g_bar "
          f"{ {f'{m}|{s}': round(v[0], 3) for (m, s), v in gbars.items()} }", flush=True)
    if args.dry_run:
        return

    from setproctitle import setproctitle
    setproctitle(f"aya_alignment:heldout_{args.tag}")
    folder.mkdir(parents=True, exist_ok=True)
    dirs = torch.load(args.calib / "directions.pt")
    tok, model = load_model()
    torch.manual_seed(SEED)
    edit = WeightEdit(model, READ / "delta_final.pt")
    r = r_hat(model.device)
    meta_common = {"label": LABEL, "stage": "heldout_stepE", "split": "heldout (83 pairs)", "pool": pool, "depth": depth,
                   "gate_depth": gd, "beta": BETA, "push": "final user turn span, no generated token",
                   "thresholds_r_hat": {m: {"file": str(f).format(d=gd), "c": thr_r[m]} for m, f in R_FILES.items()},
                   "w_and_thresholds": str(args.calib), "batch_size": args.batch_size,
                   "max_new_tokens": args.max_new_tokens, "seed": SEED}
    with (folder / f"env-{args.tag}.json").open("x") as f:
        json.dump(environment(args.out) | meta_common | {"runs": args.runs, "tag": args.tag,
                  "g_bar": {f"{m}|{s}": v for (m, s), v in gbars.items()}}, f, indent=1, default=str)

    checks, ref, lvl = {}, {}, {}
    for script, cond in runs:
        name = MODEL_OF[cond]
        edited = name == "read"
        if edited and not edit.on:
            edit.apply(1.0)
        assert edit.on == edited
        rows = get_rows(script, depth, "heldout", pool, args.limit_pairs)
        order = final_message(tok, rows, "auto" if cond == "N" else "romanized", XLIT) if script == "hi-Latn" else None
        for row, v in zip(rows, layer_scores(tok, model, rows, r, SCORE_LAYER, args.batch_size)):
            row["l19_score"] = v
        print(f"start {script} {cond} {args.setting} ({len(rows)} rows)", flush=True)

        state, handles, gate = {}, [], None
        if cond in ("Sr", "Rr"):
            for row in rows:
                row["gate_c"] = thr_r[name][script]
            handles = gate_hook(model, SCORE_LAYER, r, r, 0.0, BETA, state, None, True)
            gate = {"detector": "r_hat", "c": thr_r[name][script]}
        elif cond in ("Sw", "Rw"):
            c = thr_w[name][f"{script}|{gd}"]["c"]
            gb, src, n = gbars[(name, script)]
            for row in rows:
                row["gate_c"], row["gate_fold"] = c, 0
            w = dirs[f"w_{name}"].float().to(model.device)
            handles = oof_gate_hook(model, w[None].expand(5, -1).contiguous(), r, BETA, state, gb)
            gate = {"detector": f"w_{name} (250 non-sweep pairs)", "c_w": c, "c_w_cell": f"{script}|{gd}",
                    "push": "fixed", "g_bar": gb, "g_bar_from": src, "n_fired_ref": n}
        try:
            results = generate(tok, model, rows, args.batch_size, args.max_new_tokens, state if handles else None, order)
        finally:
            for h in handles:
                h.remove()

        meta = {**meta_common, "cond": cond, "model": name, "weights_edited": edited, "edit_layer": 16, "gate": gate,
                "final_message": ("auto" if cond == "N" else "romanized") if script == "hi-Latn" else None}
        save(folder / f"{script}-{cond}.jsonl", rows, results, cond, meta, args.max_new_tokens)

        if cond in ("Sr", "Rr") and f"path_{name}" not in checks:
            checks[f"path_{name}"] = path_check(tok, model, rows, results, order, r, args)
            print("path check", name, checks[f"path_{name}"], flush=True)
        ben = [i for i, row in enumerate(rows) if row["request_type"] == "benign"]
        if cond == "Zb":
            reps = reply_ids(tok, [results[i] for i in ben])
            ref[script] = ([rows[i] for i in ben], reps, forced(tok, model, [rows[i] for i in ben], reps))
        if edited and script in ref and script not in lvl:
            brow, reps, base = ref[script]
            lvl[script] = level1(base, forced(tok, model, brow, reps))

    if lvl:
        with (folder / f"level1-{args.tag}.json").open("x") as f:
            json.dump({"setting": args.setting, "definition": "benign prompts; Zb replies teacher-forced; KL(P_Zb || "
                       "P_Read) of the first token and ppl ratio, Read weights without gates", **lvl}, f, indent=1)
    if edit.on:
        edit.revert()
    (folder / f"checks-{args.tag}.json").write_text(json.dumps(checks, indent=1, default=str))
    failed = [k for k, v in checks.items() if "pass" in v and not v["pass"]]
    if failed:
        raise SystemExit(f"VERIFY failed: {failed}")
    print("done", folder, flush=True)


if __name__ == "__main__":
    main()

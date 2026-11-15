"""Step 5 generation (docs/prereg_step5.md). --data fresh (fresh_v1 test) | v2 (v2 held-out, 83 pairs).
--runs script:arm:depth; arms Zb, N, N+Rw, N+Sw, N+G, N+Rw_old, N+Sw_old. N normalizes hi-Latn only (IndicXlit; fresh:
runs/fresh_v1/xlit, v2: runs/normalize_v1/xlit.jsonl); on other scripts N+X runs X on the unchanged request.
w arms: oof_gate_hook with w in every slot and the fixed push g_bar (as heldout_gen.py); G: r_hat span gate (gate_hook,
span=True, beta 16). Thresholds per (script, depth): new = runs/final_step5/fit, old = fresh Step 3 (romanized) + old w.
Base-weight arms run first, then Read-weight arms, in one process. HF greedy, batch 16, max_new_tokens 2048, seed 42.
Writes <out>/<data>/<script>-<arm>-d<depth>.jsonl.
"""
import argparse
import json
import time
from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "refusal_direction"), str(HERE.parent / "multiturn"), str(HERE.parent / "fresh")]
from cache import environment  # noqa: E402
from explore_gen import oof_gate_hook  # noqa: E402
from extract import load_model  # noqa: E402
from gen import get_rows  # noqa: E402
from intervene import final_message, gate_hook, generate  # noqa: E402
from ov_common import SCORE_LAYER, r_hat  # noqa: E402
from read_pilot import BETA, READ, SEED, WeightEdit, save  # noqa: E402
from rows import fresh_rows  # noqa: E402

ARMS = ("Zb", "N", "N+Rw", "N+Sw", "N+G", "N+Rw_old", "N+Sw_old")
XLIT = {"fresh": Path("runs/fresh_v1/xlit/xlit.jsonl"), "v2": Path("runs/normalize_v1/xlit.jsonl")}
FIT = Path("runs/final_step5/fit")


def rows_of(data, script, depth):
    return fresh_rows(script, depth, "test") if data == "fresh" else get_rows(script, depth, "heldout", "v2")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", choices=["fresh", "v2"], required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=Path("runs/final_step5/gen"))
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    runs = [(s, a, int(d)) for s, a, d in (x.split(":") for x in args.runs)]
    folder = args.out / args.data
    for s, a, d in runs:
        assert a in ARMS and not (a == "N" and s != "hi-Latn"), (s, a)
        assert not (folder / f"{s}-{a}-d{d}.jsonl").exists(), (s, a, d)
    runs = [x for x in runs if "Rw" not in x[1]] + [x for x in runs if "Rw" in x[1]]
    folder.mkdir(parents=True, exist_ok=True)
    fit = json.loads((FIT / "thresholds.json").read_text())
    old = json.loads(Path("runs/fresh_v1/step3/thresholds.json").read_text())
    dnew, dold = torch.load(FIT / "directions.pt"), torch.load("runs/multiturn_v2/calib_w/directions.pt")
    tok, model = load_model()
    (folder / f"env-{time.strftime('%Y%m%dT%H%M%S')}.json").write_text(json.dumps(
        environment(folder) | {"runs": args.runs, "data": args.data, "fit": str(FIT), "xlit": str(XLIT[args.data]), "limit": args.limit},
        indent=1, default=str))
    edit, r = WeightEdit(model, READ / "delta_final.pt"), r_hat(model.device)
    for s, a, d in runs:
        model_name = "read" if "Rw" in a else "base"
        if model_name == "read" and not edit.on:
            edit.apply(1.0)
        assert edit.on == (model_name == "read")
        torch.manual_seed(SEED)
        rows = rows_of(args.data, s, d)[: args.limit]
        norm = a.startswith("N") and s == "hi-Latn"
        order = final_message(tok, rows, "auto" if norm else "romanized", XLIT[args.data]) if s == "hi-Latn" else None
        state, handles, gate = {}, [], None
        if a in ("N+Sw", "N+Rw", "N+Sw_old", "N+Rw_old"):
            new = not a.endswith("_old")
            c = (fit[model_name]["thresholds"][f"{s}|{d}"]["w"]["c"] if new else old["thresholds"][model_name][f"{s}|{d}"]["w"]["c"])
            gb = (fit[model_name]["g_bar"][s]["g_bar"] if new else old["g_bar"][f"{model_name}|{s}"]["g_bar"])
            w = (dnew if new else dold)[f"w_{model_name}"].float().to(model.device)
            for row in rows:
                row["gate_c"], row["gate_fold"] = c, 0
            handles = oof_gate_hook(model, w[None].expand(5, -1).contiguous(), r, BETA, state, gb)
            gate = {"detector": f"w_{model_name} ({'refit, normalized calibration' if new else 'old calib_w'})", "c_w": c, "g_bar": gb}
        elif a == "N+G":
            c = fit["base"]["thresholds"][f"{s}|{d}"]["r_hat"]["c"]
            for row in rows:
                row["gate_c"] = c
            handles = gate_hook(model, SCORE_LAYER, r, r, 0.0, BETA, state, None, True)
            gate = {"detector": "r_hat span gate (A1b)", "c_r": c, "beta": BETA}
        print("start", args.data, s, a, d, len(rows), flush=True)
        try:
            results = generate(tok, model, rows, 16, 2048, state if handles else None, order)
        finally:
            for h in handles:
                h.remove()
        save(folder / f"{s}-{a}-d{d}.jsonl", rows, results, a,
             {"label": "Step 5 (prereg_step5.md)", "arm": a, "model": model_name, "normalized": norm, "gate": gate}, 2048)
    if edit.on:
        edit.revert()


if __name__ == "__main__":
    main()

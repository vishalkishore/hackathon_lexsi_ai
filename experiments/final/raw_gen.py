"""ADDED experiment generation (docs/prereg_raw.md). RAW input (no IndicXlit). fresh_v1 TEST.
--runs script:arm:depth; arms Zb, Gate-raw, Read-raw, Read+Gate-raw. Gate arms: oof_gate_hook with w in every slot and the
fixed push g_bar (as final_gen.py), thresholds per (script, depth) from runs/raw_repair/fit/thresholds.json; Read arms
apply delta_read_raw.pt (layer 16). Base-weight arms first, then Read-weight arms. HF greedy, batch 16, 2048 tokens,
seed 42; hi-Latn batch order = the romanized order (as final_gen.py). Writes runs/raw_repair/gen/<script>-<arm>-d<depth>.jsonl.
"""
import argparse
import json
import time
from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]
from final_gen import BETA, SEED, environment, final_message, generate, load_model, oof_gate_hook, r_hat, rows_of, save  # noqa: E402
from read_pilot import WeightEdit  # noqa: E402

FIT = Path("runs/raw_repair/fit")
ARMS = ("Zb", "Gate-raw", "Read-raw", "Read+Gate-raw")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=Path("runs/raw_repair/gen"))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--fit", type=Path, default=FIT)
    args = ap.parse_args()
    runs = [(s, a, int(d)) for s, a, d in (x.split(":") for x in args.runs)]
    for s, a, d in runs:
        assert a in ARMS and s in ("hi-Latn", "en-hi", "en"), (s, a)
        assert not (args.out / f"{s}-{a}-d{d}.jsonl").exists(), (s, a, d)
    runs = [x for x in runs if "Read" not in x[1]] + [x for x in runs if "Read" in x[1]]
    args.out.mkdir(parents=True, exist_ok=True)
    fit = json.loads((args.fit / "thresholds.json").read_text())
    dirs = torch.load(args.fit / "directions.pt")
    tok, model = load_model()
    (args.out / f"env-{time.strftime('%Y%m%dT%H%M%S')}.json").write_text(json.dumps(
        environment(args.out) | {"runs": args.runs, "fit": str(args.fit), "limit": args.limit, "prereg": "docs/prereg_raw.md"}, indent=1, default=str))
    edit, r = WeightEdit(model, args.fit / "delta_read_raw.pt"), r_hat(model.device)
    for s, a, d in runs:
        if "Read" in a and not edit.on:
            edit.apply(1.0)
        assert edit.on == ("Read" in a)
        torch.manual_seed(SEED)
        rows = rows_of("fresh", s, d)[: args.limit]
        order = final_message(tok, rows, "romanized") if s == "hi-Latn" else None
        state, handles, gate = {}, [], None
        if "Gate" in a:
            key = "read_gate_raw" if "Read" in a else "gate_raw"
            c, gb = fit[key]["thresholds"][f"{s}|{d}"]["w"]["c"], fit[key]["g_bar"][s]["g_bar"]
            w = dirs["w_read_raw" if "Read" in a else "w_raw_fresh"].float().to(model.device)
            for row in rows:
                row["gate_c"], row["gate_fold"] = c, 0
            handles = oof_gate_hook(model, w[None].expand(5, -1).contiguous(), r, BETA, state, gb)
            gate = {"detector": key, "c_w": c, "g_bar": gb}
        print("start", s, a, d, len(rows), flush=True)
        try:
            results = generate(tok, model, rows, 16, 2048, state if handles else None, order)
        finally:
            for h in handles:
                h.remove()
        save(args.out / f"{s}-{a}-d{d}.jsonl", rows, results, a,
             {"label": "ADDED raw repair (prereg_raw.md)", "arm": a, "read_edit": "Read" in a, "normalized": False, "gate": gate}, 2048)
    if edit.on:
        edit.revert()


if __name__ == "__main__":
    main()

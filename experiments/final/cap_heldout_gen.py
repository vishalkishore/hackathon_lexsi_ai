"""Capability check for the exploration-set main method (docs/prereg_capability_heldout.md): arm Rw = Weight edit
(delta_final.pt, layer 16) + Gated steer (w): w_read from runs/multiturn_v2/calib_w, c_w = read en|0 threshold,
push beta * g_bar (read; en uses the hi-Latn value, as heldout_gen.g_bar). English single turn, same rows, system
prompt and decoding as cap_gen.py, whose Zb outputs (runs/final_step5/capability) are the Base arm.
Writes runs/heldout_capability/<set>-Rw.jsonl."""
import argparse
import json
import time
from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "refusal_direction"), str(HERE.parent / "multiturn"), str(HERE.parent / "fresh")]
from cap_gen import rows_of  # noqa: E402
from cache import environment  # noqa: E402
from explore_gen import oof_gate_hook  # noqa: E402
from extract import load_model  # noqa: E402
from heldout_gen import g_bar  # noqa: E402
from intervene import generate  # noqa: E402
from ov_common import r_hat  # noqa: E402
from read_pilot import BETA, READ, SEED, WeightEdit, save  # noqa: E402

CALIB = Path("runs/multiturn_v2/calib_w")
EXPLORE = Path("runs/multiturn_v2/m2_explore")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sets", nargs="+", choices=["gsm8k", "xstest"], required=True)
    ap.add_argument("--out", type=Path, default=Path("runs/heldout_capability"))
    ap.add_argument("--limit", type=int, help="smoke tests only")
    ap.add_argument("--arm", choices=["Rw", "Sw"], default="Rw", help="Sw: base weights, calib_w w_base / base c_w / base g_bar (gate alone)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for s in args.sets:
        assert not (args.out / f"{s}-{args.arm}.jsonl").exists(), s
    name = "read" if args.arm == "Rw" else "base"
    c = json.loads((CALIB / "thresholds.json").read_text())[name]["en|0"]["c"]
    gb, src, n = g_bar(EXPLORE, name, "en")
    tok, model = load_model()
    edit = WeightEdit(model, READ / "delta_final.pt")
    if name == "read":
        edit.apply(1.0)
    r = r_hat(model.device)
    w = torch.load(CALIB / "directions.pt")[f"w_{name}"].float().to(model.device)
    gate = {"detector": f"w_{name} (calib_w, 250 non-sweep pairs)", "c_w": c, "c_w_cell": "en|0", "g_bar": gb, "g_bar_from": src,
            "n_fired_ref": n, "weights_edited": name == "read", "edit_layer": 16}
    (args.out / f"env-{time.strftime('%Y%m%dT%H%M%S')}.json").write_text(json.dumps(
        environment(args.out) | {"sets": args.sets, "arm": args.arm, "limit": args.limit, "gate": gate}, indent=1, default=str))
    for s in args.sets:
        torch.manual_seed(SEED)
        rows, state = rows_of(s)[: args.limit], {}
        for row in rows:
            row["gate_c"], row["gate_fold"] = c, 0
        handles = oof_gate_hook(model, w[None].expand(5, -1).contiguous(), r, BETA, state, gb)
        print("start", s, len(rows), flush=True)
        try:
            results = generate(tok, model, rows, 16, 2048, state)
        finally:
            for h in handles:
                h.remove()
        save(args.out / f"{s}-{args.arm}.jsonl", rows, results, args.arm, {"label": "capability (prereg_capability_heldout.md; Sw: prereg_controls.md Step 3)", "arm": args.arm, "gate": gate}, 2048)
    if name == "read":
        edit.revert()


if __name__ == "__main__":
    main()

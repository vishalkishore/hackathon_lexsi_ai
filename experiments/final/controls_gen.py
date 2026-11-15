"""Controls Step 2 generation, AMENDED (docs/prereg_controls.md, amendment 2026-11-15). v2 held-out, hi-Latn, RAW
romanized input, --depths 8 0. Settings as heldout_gen.py (oof_gate_hook, final-user-turn push, beta 16, batch 16,
seed 42, 2048 tokens, romanized batch order). Arms (<base> = Rw or Sw):
  RD<s>-<base>  detector v = unit Gaussian (seed s), c_v from runs/controls/controls_cv.json (controls_cv.py), push
                beta * g_bar * r_hat with <base>'s g_bar
  RP<s>-<base>  <base>'s calib_w w and c_w; push beta * g_bar * u, u = unit Gaussian (seed s + 1000)
  check-Rw      reproduction check: the real Rw gate on the first 16 rows (one batch); must equal the stored
                heldout_E Rw replies, else stop.
Rw arms use the weight edit (delta_final.pt) and read-model w / c / g_bar; Sw arms base weights.
Writes runs/controls/gen/hi-Latn-<arm>-d<depth>.jsonl."""
import argparse
import json
import time
from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "multiturn")]
from final_gen import BETA, READ, SEED, WeightEdit, environment, final_message, generate, load_model, oof_gate_hook, r_hat, rows_of, save  # noqa: E402
from heldout_gen import g_bar  # noqa: E402

CALIB, EXPLORE, HELD = Path("runs/multiturn_v2/calib_w"), Path("runs/multiturn_v2/m2_explore"), Path("runs/multiturn_v2/heldout_E")
MODEL = {"Rw": "read", "Sw": "base"}


def unit(seed, n=4096):
    v = torch.randn(n, generator=torch.Generator().manual_seed(seed))
    return v / v.norm()


def parse(a):
    if a == "check-Rw":
        return "check", None, "Rw"
    kind, rest = a[:2], a[2:]
    seed, base = rest.split("-")
    assert kind in ("RD", "RP") and base in MODEL, a
    return kind, int(seed), base


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--depths", nargs="+", type=int, default=[8, 0])
    ap.add_argument("--out", type=Path, default=Path("runs/controls/gen"))
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    arms = [(a, *parse(a)) for a in args.arms]
    for a, *_ in arms:
        for d in args.depths:
            assert not (args.out / f"hi-Latn-{a}-d{d}.jsonl").exists(), (a, d)
    arms = [x for x in arms if x[3] == "Sw"] + [x for x in arms if x[3] == "Rw"]  # base weights first
    args.out.mkdir(parents=True, exist_ok=True)
    thr, dirs = json.loads((CALIB / "thresholds.json").read_text()), torch.load(CALIB / "directions.pt")
    cv = json.loads(Path("runs/controls/controls_cv.json").read_text())
    tok, model = load_model()
    dev = model.device
    (args.out / f"env-{time.strftime('%Y%m%dT%H%M%S')}.json").write_text(json.dumps(
        environment(args.out) | {"arms": args.arms, "depths": args.depths, "calib": str(CALIB), "limit": args.limit,
                                 "prereg": "docs/prereg_controls.md (amended)"}, indent=1, default=str))
    edit, r = WeightEdit(model, READ / "delta_final.pt"), r_hat(dev)
    for a, kind, seed, base in arms:
        name = MODEL[base]
        if name == "read" and not edit.on:
            edit.apply(1.0)
        assert edit.on == (name == "read")
        gb, src, _ = g_bar(EXPLORE, name, "hi-Latn")
        w = dirs[f"w_{name}"].float().to(dev)
        for d in args.depths:
            torch.manual_seed(SEED)
            rows = rows_of("v2", "hi-Latn", d)
            order = final_message(tok, rows, "romanized")
            if kind == "check":
                order = order[:16]
            elif args.limit:
                order = order[: args.limit]
            c_w = thr[name][f"hi-Latn|{d}"]["c"]
            if kind == "RD":
                det, push, c = unit(seed).to(dev), r, cv[f"RD{seed}|{name}|{d}"]["c"]
            elif kind == "RP":
                det, push, c = w, unit(seed + 1000).to(dev), c_w
            else:
                det, push, c = w, r, c_w
            for row in rows:
                row["gate_c"], row["gate_fold"] = c, 0
            state = {}
            handles = oof_gate_hook(model, det[None].expand(5, -1).contiguous(), push, BETA, state, gb)
            print("start", a, d, len(order), flush=True)
            try:
                results = generate(tok, model, rows, 16, 2048, state, order)
            finally:
                for h in handles:
                    h.remove()
            if kind == "check":
                stored = {x["item_id"]: x["response"] for x in map(json.loads, open(HELD / f"v2d{d}" / "hi-Latn-Rw.jsonl"))}
                same = [results[i][0] == stored[rows[i]["item_id"]] for i in order]
                (args.out / f"check-Rw-d{d}.json").write_text(json.dumps({"identical": sum(same), "n": len(same)}))
                print("check", d, sum(same), "/", len(same), flush=True)
                assert all(same), f"reproduction check failed d{d}: {sum(same)}/{len(same)}"
                continue
            keep = [i for i in order]
            save(args.out / f"hi-Latn-{a}-d{d}.jsonl", [rows[i] for i in keep], [results[i] for i in keep], a,
                 {"label": "controls amended (prereg_controls.md)", "arm": a, "control": kind, "seed": seed, "base_arm": base,
                  "weights_edited": name == "read", "detector": "random" if kind == "RD" else f"w_{name} (calib_w)",
                  "push": "random" if kind == "RP" else "r_hat", "c": c, "g_bar": gb, "g_bar_from": src}, 2048)
    if edit.on:
        edit.revert()


if __name__ == "__main__":
    main()

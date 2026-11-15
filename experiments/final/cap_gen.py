"""Capability side-check generation (docs/prereg_capability.md). --set gsm8k | xstest; --arms Zb N+Sw.
English single turn; N+Sw uses the en|0 threshold and en g_bar of runs/final_step5/fit (base weights).
Writes runs/final_step5/capability/<set>-<arm>.jsonl."""
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
from generate import SYSTEM  # noqa: E402
from intervene import generate  # noqa: E402
from ov_common import r_hat  # noqa: E402
from read_pilot import BETA, SEED, save  # noqa: E402

FIT = Path("runs/final_step5/fit")
REV = {"gsm8k": "740312add88f781978c0658806c59bc2815b9866", "xstest": "f1d713187c61b6ae64e602d74f0b3d812cc2e8e8"}
SUFFIX = "\nEnd with the final answer as a number on the last line."


def rows_of(name):
    from datasets import load_dataset
    if name == "gsm8k":
        d = load_dataset("openai/gsm8k", "main", split="test", revision=REV[name])
        xs = [{"item_id": f"gsm{i}", "request": x["question"] + SUFFIX, "request_type": "benign",
               "answer": x["answer"].split("####")[-1].strip().replace(",", "")} for i, x in enumerate(d)]
    else:
        d = load_dataset("walledai/XSTest", split="test", revision=REV[name])
        xs = [{"item_id": f"xs{i}", "request": x["prompt"], "request_type": "harmful" if x["label"] == "unsafe" else "benign",
               "xstest_type": x["type"]} for i, x in enumerate(d)]
    return [{**x, "group_id": x["item_id"], "pair_id": x["item_id"], "request_english": x["request"], "switch": "en", "depth": 0,
             "key": f"{name}|{x['item_id']}", "single_turn": True,
             "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": x["request"]}]} for x in xs]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sets", nargs="+", choices=["gsm8k", "xstest"], required=True)
    ap.add_argument("--arms", nargs="+", choices=["Zb", "N+Sw"], required=True)
    ap.add_argument("--out", type=Path, default=Path("runs/final_step5/capability"))
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for s in args.sets:
        for a in args.arms:
            assert not (args.out / f"{s}-{a}.jsonl").exists(), (s, a)
    fit = json.loads((FIT / "thresholds.json").read_text())
    tok, model = load_model()
    (args.out / f"env-{time.strftime('%Y%m%dT%H%M%S')}.json").write_text(json.dumps(
        environment(args.out) | {"sets": args.sets, "arms": args.arms, "fit": str(FIT), "limit": args.limit, "rev": REV}, indent=1, default=str))
    r = r_hat(model.device)
    w = torch.load(FIT / "directions.pt")["w_base"].float().to(model.device)
    c, gb = fit["base"]["thresholds"]["en|0"]["w"]["c"], fit["base"]["g_bar"]["en"]["g_bar"]
    for s in args.sets:
        for a in args.arms:
            torch.manual_seed(SEED)
            rows, state, handles, gate = rows_of(s)[: args.limit], {}, [], None
            if a == "N+Sw":
                for row in rows:
                    row["gate_c"], row["gate_fold"] = c, 0
                handles = oof_gate_hook(model, w[None].expand(5, -1).contiguous(), r, BETA, state, gb)
                gate = {"detector": "w_base (refit, normalized calibration)", "c_w": c, "g_bar": gb, "threshold": "en|0"}
            print("start", s, a, len(rows), flush=True)
            try:
                results = generate(tok, model, rows, 16, 2048, state if handles else None)
            finally:
                for h in handles:
                    h.remove()
            save(args.out / f"{s}-{a}.jsonl", rows, results, a, {"label": "capability (prereg_capability.md)", "arm": a, "gate": gate}, 2048)


if __name__ == "__main__":
    main()

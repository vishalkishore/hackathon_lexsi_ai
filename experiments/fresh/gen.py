"""Fresh normalization runs (NOTES.md, FRESH NORMALIZATION, pre-registered): base weights, no gate, 198 test pairs.
--runs script:cond:depth with cond Zb (as written), N (hi-Latn only: final user message = IndicXlit Devanagari) or
Zl (hi-Latn, depth > 0: zero edit with the v2 prefix rendered in romanized Hindi, runs/fresh_v1/prefix_latn).
HF greedy, batch 16, max_new_tokens 2048, seed 42; hi-Latn batch order fixed by the romanized prompts
(intervene.final_message), so Zb and N share batch membership. Writes <out>/<script>-<cond>-d<depth>.jsonl.
"""
import argparse
import json
from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "refusal_direction"))
sys.path.insert(0, str(HERE.parent / "multiturn"))
sys.path.insert(0, str(HERE))
from cache import environment  # noqa: E402
from extract import load_model  # noqa: E402
from intervene import final_message, generate  # noqa: E402
from read_pilot import SEED, save  # noqa: E402
from rows import fresh_rows  # noqa: E402
sys.path.insert(0, str(HERE.parent / "codeswitch"))
from generate import messages  # noqa: E402

XLIT = Path("runs/fresh_v1/xlit/xlit.jsonl")
PREFIX_LATN = Path("runs/fresh_v1/prefix_latn/prefixes.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--limit", type=int, help="smoke test: first N rows")
    args = ap.parse_args()
    runs = [tuple(x.split(":")) for x in args.runs]
    for script, cond, depth in runs:
        assert cond in ("Zb", "N", "Zl") and not (cond in ("N", "Zl") and script != "hi-Latn"), (script, cond)
        assert not (cond == "Zl" and depth == "0"), "Zl needs a prefix"
        assert not (args.out / f"{script}-{cond}-d{depth}.jsonl").exists(), (script, cond, depth)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"env-{'_'.join(a + b + c for a, b, c in runs)}.json").write_text(
        json.dumps(environment(args.out) | {"runs": args.runs, "xlit": str(XLIT), "limit": args.limit}, indent=1, default=str))
    tok, model = load_model()
    for script, cond, depth in runs:
        torch.manual_seed(SEED)
        rows = fresh_rows(script, int(depth), "test")[: args.limit]
        if cond == "Zl":
            pref = json.loads(PREFIX_LATN.read_text())
            for r in rows:
                r["opener_turns"] = pref[json.dumps(r["opener_turns"], ensure_ascii=False)]["latn"]
                r["messages"] = messages(r)
                r["prefix_language"] = "hi-Latn"
        order = final_message(tok, rows, "auto" if cond == "N" else "romanized", XLIT) if script == "hi-Latn" else None
        print("start", script, cond, depth, len(rows), flush=True)
        results = generate(tok, model, rows, 16, 2048, None, order)
        meta = {"label": "fresh normalization (pre-registered)", "cond": cond, "weights_edited": False,
                "final_message": ("auto" if cond == "N" else "romanized") if script == "hi-Latn" else None}
        save(args.out / f"{script}-{cond}-d{depth}.jsonl", rows, results, cond, meta, 2048)


if __name__ == "__main__":
    main()

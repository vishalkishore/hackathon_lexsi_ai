"""SafeTune 0.1.10 r_hat ablation: RefusalDirectionModel(mode="ablate") hooks every decoder layer; the same SafeTune
_hook is also registered on the embeddings, so the edit matches ayasafe.mechanism.ablation (r_hat projected out after the
embeddings and every decoder layer). Rows: the 83 held-out English harmful single-turn prompts (ayasafe.data.correction);
HF greedy, batch 16, 512 new tokens (ayasafe.model.generate / save). Label with the refusal-only judge afterwards.
  PYTHONPATH=submission/repo python submission/repo/kits/safetune/ablation.py --correction-heretic <pairs_main_prefixes.parquet> --prefixes <prefixes.parquet>"""
import argparse
from pathlib import Path

import torch
from safetune.interventions.steer.refusal_direction import RefusalDirectionModel

from ayasafe import data, model as M


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--correction-heretic", type=Path, default=Path("submission/hf_datasets/aya8b-hindi-safety-prompts/data/heretic/pairs_main_prefixes.parquet"))
    ap.add_argument("--prefixes", type=Path, default=Path("submission/hf_datasets/aya8b-hindi-safety-prompts/data/prefixes.parquet"))
    ap.add_argument("--r-hat", type=Path, default=Path("runs/refusal_direction_v1/fit/directions.pt"))
    ap.add_argument("--layer", type=int, default=19)
    ap.add_argument("--out", type=Path, default=Path("runs/kits/ablation/safetune-ablate-L19.jsonl"))
    args = ap.parse_args()
    rows = [r for r in data.correction({"heretic": args.correction_heretic, "prefixes": args.prefixes}, "en", 0, "heldout")
            if r["request_type"] == "harmful"]
    for r in rows:
        r["single_turn"] = True  # guard_judge reads single-turn rows as user + assistant only
    assert len(rows) == 83, len(rows)
    tok, model = M.load()
    r = torch.load(args.r_hat)["r_hat"]["en"][args.layer, 0].to(model.device)
    rd = RefusalDirectionModel(model, r, mode="ablate")
    rd.install()
    emb = model.model.embed_tokens.register_forward_hook(rd._hook)
    try:
        res = M.generate(tok, model, rows, 16, 512)
    finally:
        emb.remove()
        rd.remove()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    M.save(args.out, rows, res, f"safetune-ablate-L{args.layer}", {"kit": "safetune 0.1.10 RefusalDirectionModel + embedding hook", "layer": args.layer}, 512)


if __name__ == "__main__":
    main()

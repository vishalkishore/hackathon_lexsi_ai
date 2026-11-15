"""Capability side-check generations: GSM8K (first n test items) and XSTest, single turn, English (normalization is the
identity), per arm; XSTest replies get v4 labels. report.capability scores them."""
import json
from pathlib import Path

import torch

from . import data, judge, model as M, repair

SUFFIX = "\nEnd with the final answer as a number on the last line."


def capability(cfg):
    """GSM8K (first n test items) and XSTest prompts, single turn, for the arms in cfg['arms'] (base or Read weights, gate
    thresholds of en|0). GSM8K: last number in the reply == gold answer. XSTest replies are labelled with judge.label
    (refusal = REFUSED or REDIRECTED); the safe/unsafe type comes from the dataset."""
    from datasets import load_dataset
    out = Path(cfg["out"])
    out.mkdir(parents=True, exist_ok=True)
    gsm = load_dataset("openai/gsm8k", "main", split="test", revision=cfg["gsm8k_revision"]).select(range(cfg["gsm8k_n"]))
    xs = load_dataset(cfg["xstest"], split=cfg["xstest_split"], revision=cfg["xstest_revision"])
    sets = {"gsm8k": [{"item_id": f"gsm{i}", "group_id": f"gsm{i}", "request": x["question"] + SUFFIX, "request_english": x["question"],
                       "request_type": "benign", "answer": x["answer"].split("####")[-1].strip().replace(",", "")} for i, x in enumerate(gsm)],
            "xstest": [{"item_id": f"xs{i}", "group_id": f"xs{i}", "request": x["prompt"], "request_english": x["prompt"],
                        "request_type": "harmful" if x["label"] == "unsafe" else "benign", "xstest_type": x["type"]} for i, x in enumerate(xs)]}
    tok, model = M.load()
    r = repair.r_hat(cfg["r_hat"]).to(model.device)
    fit_dir = Path(cfg["fit"])
    th = json.loads((fit_dir / "thresholds.json").read_text())
    w = {m: torch.load(fit_dir / "directions.pt")[f"w_{m}"].float().to(model.device) for m in ("base", "read")}
    edit = M.WeightEdit(model, cfg["read_delta"])
    for arm in cfg["arms"]:
        weights = "read" if "Rw" in arm else "base"
        edit.apply(1.0) if weights == "read" else edit.revert()
        for name, rows in sets.items():
            rows = [{**x, "opener_turns": [], "key": f"{name}|{arm}|{x['item_id']}"} for x in rows]
            state, hooks = None, []
            if arm in ("N+G", "N+Sw", "N+Rw"):
                c = th["base" if arm == "N+G" else weights]["thresholds"]["en|0"]["r_hat" if arm == "N+G" else "w"]["c"]
                for x in rows:
                    x["gate_c"] = c
                state = {}
                hooks = [M.span_gate(model, r if arm == "N+G" else w[weights], r, cfg["beta"], state,
                                     None if arm == "N+G" else th[weights]["g_bar"]["en"]["g_bar"])]
            try:
                res = M.generate(tok, model, rows, 16, cfg["max_new_tokens"], state)
            finally:
                for h in hooks:
                    h.remove()
            M.save(out / f"{name}-{arm}.jsonl", rows, res, arm, {"arm": arm, "weights": weights}, cfg["max_new_tokens"])
    edit.revert()
    xs_rows = [x for arm in cfg["arms"] for x in data.jl(out / f"xstest-{arm}.jsonl")]
    judge.label(xs_rows, out / "labels.jsonl")

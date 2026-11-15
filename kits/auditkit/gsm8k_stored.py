"""AuditKIT 1.1.3 on STORED GSM8K replies (no generation): auditkit.evaluate with a CallableModel that returns the stored
reply for each prompt and an auditkit.scorer = ayasafe.report.gsm8k_correct (last number in the reply == gold);
auditkit.compare(base, main).significance() for the paired bootstrap by sample id.
  PYTHONPATH=submission/repo python submission/repo/kits/auditkit/gsm8k_stored.py --base <gsm8k-Zb.jsonl> --main <gsm8k-Rw.jsonl>
Reproduces: Base 80.06, main method 80.67 (Δ +0.61, p 0.58)."""
import argparse
import json
from pathlib import Path

import auditkit as ak
from auditkit.model import CallableModel

from ayasafe.report import gsm8k_correct


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, default=Path("runs/final_step5/capability/gsm8k-Zb.jsonl"))
    ap.add_argument("--main", type=Path, default=Path("runs/heldout_capability/gsm8k-Rw.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("runs/kits/auditkit_gsm8k.json"))
    args = ap.parse_args()
    runs = {"Base": args.base, "main": args.main}
    rows = {k: [json.loads(x) for x in open(p)] for k, p in runs.items()}
    assert [r["item_id"] for r in rows["Base"]] == [r["item_id"] for r in rows["main"]]
    samples = [ak.Sample(input=r["request"], target=r["answer"], id=r["item_id"], task="gsm8k") for r in rows["Base"]]

    @ak.scorer(direction=ak.Direction.MAXIMIZE, name="last_number_equals_gold")
    def last_number(sample, output):
        return float(gsm8k_correct({"response": output, "answer": sample.target}))

    res, out = {}, {}
    for k, rs in rows.items():
        by_id = {r["item_id"]: r["response"] for r in rs}
        by_input = {s.input: by_id[s.id] for s in samples}
        assert len(by_input) == len(samples), "duplicate prompts"
        # explicit name: AuditKIT's disk cache keys on the model name; two anonymous callables can collide
        model = CallableModel(lambda prompts, m=by_input: [m[p] for p in prompts], name=f"stored-replies:{runs[k]}")
        res[k] = ak.evaluate(samples, model, scorers=last_number, experiment_name=f"gsm8k-{k}")
        out[k] = {"auditkit": res[k].headline, "ours_pct": 100 * sum(gsm8k_correct(r) for r in rs) / len(rs), "n": len(rs)}
    out["significance"] = ak.compare(res["Base"], res["main"]).significance("last_number_equals_gold")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()

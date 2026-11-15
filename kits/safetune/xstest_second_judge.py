"""SafeTune 0.1.10 XSTest over-refusal metric (safetune.core.eval.metrics.safety.OverRefusalMetric, the safety suite's
keyword refusal judge) applied to STORED XSTest replies, as a second judge next to the v4 labels (ayasafe.judge cache).
Paired bootstrap / McNemar: ayasafe.stats.
  PYTHONPATH=submission/repo python submission/repo/kits/safetune/xstest_second_judge.py --base <xstest-Zb.jsonl> --main <xstest-Rw.jsonl>
Reproduces: safe over-refusal Base 4.8 -> main 11.6 (Δ +6.8 [+3.2, +10.8]); v4 10.0 -> 18.0."""
import argparse
import json
from pathlib import Path

from safetune.core.eval.metrics.safety import OverRefusalMetric, _is_refusal

from ayasafe import judge, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, default=Path("runs/final_step5/capability/xstest-Zb.jsonl"))
    ap.add_argument("--main", type=Path, default=Path("runs/heldout_capability/xstest-Rw.jsonl"))
    ap.add_argument("--labels", type=Path, default=Path("runs/rescore_v4/cache.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("runs/kits/safetune_xstest.json"))
    args = ap.parse_args()
    cache, out, flags = judge.load_cache(args.labels), {}, {}
    for k, p in (("Base", args.base), ("main", args.main)):
        rows = [json.loads(x) for x in open(p)]
        for rt, tag in (("benign", "safe"), ("harmful", "unsafe")):
            sub = [r for r in rows if r["request_type"] == rt]
            st = OverRefusalMetric().compute([r["response"] for r in sub], [])["over_refusal_rate"]
            v4 = [cache.get(judge.key(r)) for r in sub]
            v4r = [y["label"] in ("REFUSED", "REDIRECTED") for y in v4 if y and y["label"]]
            flags[(k, tag)] = {r["item_id"]: int(_is_refusal(r["response"])) for r in sub}
            out[f"{k} | XSTest {tag} | refusal %"] = {"safetune_keyword": 100 * st, "ours_v4": 100 * sum(v4r) / len(v4r),
                                                       "n": len(sub), "n_v4_labelled": len(v4r)}
    for tag in ("safe", "unsafe"):
        a, b = flags[("main", tag)], flags[("Base", tag)]
        d, lo, hi, n = stats.boot(a, b)
        out[f"delta main - Base | XSTest {tag} | safetune_keyword"] = {"delta": d, "ci": [lo, hi], "n": n, "p_mcnemar": stats.mcnemar(a, b)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

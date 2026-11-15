"""Cost per 1,000 replies: generation on a rented RTX A6000 (Vast.ai) vs labelling with the gpt-6-luna API.
Prices (given by the user, 2026-11-15): Vast.ai $0.35 per GPU-hour; gpt-6-luna $0.10 / $0.50 per 1M input / output tokens.
Generation time: gaps between saved files of the sequential controls queues (runs/controls/gen, 166 replies each; HF greedy,
batch 16, up to 2,048 new tokens); runs with gaps > 2,000 s (queue pauses or a shared GPU) give the slow end of the range only.
Judge tokens: the filled v4 prompts of the same replies, tokenized with tiktoken o200k_base (an estimate: the deployment's own
tokenizer is not exposed). Output: the two-line answer (12 tokens) as the low end; the 512-token cap (hidden reasoning tokens
are billed as output) as the high end. Writes runs/cost/cost.csv."""
import glob
import json
import statistics
import sys
from pathlib import Path

import pandas as pd
import tiktoken

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root, for ayasafe
from ayasafe import judge  # noqa: E402

GPU_HR, IN_M, OUT_M, CAP = 0.35, 0.10, 0.50, 512
files = sorted(glob.glob("runs/controls/gen/hi-Latn-R*.jsonl"))
gaps = []
for arm in ("Rw", "Sw"):
    t = sorted(Path(f).stat().st_mtime for f in files if f"-{arm}-" in f)
    gaps += [b - a for a, b in zip(t, t[1:])]
n_run = 166
steady = [g for g in gaps if g <= 2000]
slow = max(g for g in gaps if g <= 4000)
enc = tiktoken.get_encoding("o200k_base")
tin = statistics.mean(len(enc.encode(judge.filled(json.loads(l)))) for f in files for l in open(f))
tout = len(enc.encode("LABEL: REDIRECTED\nHARMFUL_HELP: NO"))
cache = [json.loads(l) for l in open("runs/rescore_v4/cache.jsonl")]
n_luna = sum(c.get("backend") == "luna" for c in cache)
gen = lambda s: 1000 / (n_run / s * 3600) * GPU_HR  # noqa: E731
lab = lambda o: 1000 * (tin * IN_M + o * OUT_M) / 1e6  # noqa: E731
rows = [{"item": "generation, Aya 8B on RTX A6000 (Vast.ai)", "seconds_per_166": statistics.median(steady), "replies_per_hour": n_run / statistics.median(steady) * 3600,
         "usd_per_1k": gen(statistics.median(steady)), "usd_per_1k_low": gen(min(steady)), "usd_per_1k_high": gen(slow), "runs": len(steady)},
        {"item": "labelling, gpt-6-luna API", "input_tokens": tin, "output_tokens_low": tout, "output_tokens_high": CAP,
         "usd_per_1k_low": lab(tout), "usd_per_1k_high": lab(CAP), "luna_labels": n_luna,
         "usd_total_low": n_luna * lab(tout) / 1000, "usd_total_high": n_luna * lab(CAP) / 1000},
        {"item": "all GPU work, about 240 GPU-hours", "usd_total": 240 * GPU_HR}]
for r in rows:
    r.update(price_gpu_hour=GPU_HR, price_in_1m=IN_M, price_out_1m=OUT_M)
Path("runs/cost").mkdir(exist_ok=True)
pd.DataFrame(rows).to_csv("runs/cost/cost.csv", index=False)
print(pd.DataFrame(rows).T.to_string())

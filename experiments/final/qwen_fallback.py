"""Low-memory Qwen fallback (docs/prereg_controls.md Step 3): label the replies that have no v4 label after luna.
Same prompt (rescore_v4.filled) and decoding as rescore_v4 (Qwen3.5-27B-FP8, temperature 0, top_p 1, max_tokens 32,
seed 0, no thinking); only memory differs: --cpu-offload-gb, --mem, --max-len. Appends to the v4 cache.
  uv run --no-sync python circuit/final/qwen_fallback.py --dirs runs/final_step5/capability ..."""
import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "rescore")]
import rescore_v4 as R  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--cpu-offload-gb", type=float, default=16)
    ap.add_argument("--mem", type=float, default=0.55)
    ap.add_argument("--max-len", type=int, default=8192)
    args = ap.parse_args()
    R.SETS = tuple(args.dirs)
    cache = R.load_cache()
    todo = {}
    for p in R.files():
        for r in map(json.loads, open(p)):
            q = R.filled(r)
            h = R.hkey(q)
            if h not in cache or cache[h]["label"] is None:
                todo[h] = q
    print(f"unlabelled before: {len(todo)}", flush=True)
    if todo:
        from vllm import LLM, SamplingParams
        llm = LLM(model=R.QWEN, max_model_len=args.max_len, gpu_memory_utilization=args.mem, cpu_offload_gb=args.cpu_offload_gb,
                  seed=0, max_num_seqs=8, enforce_eager=True)
        items = list(todo.items())
        res = llm.chat([[{"role": "user", "content": q}] for _, q in items],
                       SamplingParams(temperature=0.0, top_p=1.0, max_tokens=32, seed=0), chat_template_kwargs={"enable_thinking": False})
        with R.CACHE.open("a") as f:
            for (h, _), o in zip(items, res):
                lab, hh = R.parse(o.outputs[0].text)
                y = {"h": h, "label": lab, "harmful_help": hh, "backend": "qwen", "raw": o.outputs[0].text[:200],
                     "memory": {"cpu_offload_gb": args.cpu_offload_gb, "mem": args.mem, "max_len": args.max_len}}
                cache[h] = y
                f.write(json.dumps(y) + "\n")
    print("unlabelled after:", sum(cache[h]["label"] is None for h in todo), flush=True)


if __name__ == "__main__":
    main()

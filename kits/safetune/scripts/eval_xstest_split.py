
import argparse
import json
from collections import Counter
from pathlib import Path

import safetune
from safetune.data.loaders import load_xstest
from safetune.evaluate import evaluate
from safetune.runner.utils.model_utils import load_model
from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    default="CohereLabs/aya-expanse-8b",
)
parser.add_argument(
    "--output",
    default="results/baseline/safety/xstest/xstest_split.json",
)
parser.add_argument("--batch-size", type=int, default=4)
args = parser.parse_args()

# 1. Verify dataset labels
dataset = load_xstest()
counts = Counter(
    row["label"].strip().lower()
    for row in dataset
)
print("Dataset labels:", counts)

assert counts["safe"] == 250, counts
assert counts["unsafe"] == 200, counts

# 2. Load model once
model = load_model(args.model)
tokenizer = AutoTokenizer.from_pretrained(args.model)
model.eval()


results = {}

# 3. Evaluate each subset independently
for label in ["safe", "unsafe"]:

    safetune.configure(
        datasets={
            "xstest": {
                "where": {"label": label}
            }
        }
    )

    print(f"\nEvaluating XSTest: {label}")

    result = evaluate(
        model,
        tokenizer=tokenizer,
        benchmarks=["xstest"],
        judge="wildguard",
        batch_size=args.batch_size,
        max_new_tokens=512,
        strict=True,
    )["xstest"]

    assert result["n_evaluated"] == counts[label], result

    results[label] = {
        "n": result["n_evaluated"],
        "refusal_rate": result["refusal_rate"],
        "non_refusal_rate": result["asr"],
    }

    print(json.dumps(results[label], indent=2))

# 4. Report the correct metrics
safe_rate = results["safe"]["refusal_rate"]
unsafe_rate = results["unsafe"]["refusal_rate"]

summary = {
    "model": args.model,
    "judge": "wildguard",
    "safe_over_refusal_rate": safe_rate,
    "unsafe_refusal_rate": unsafe_rate,
    "subsets": results,
}

print("\n===== XSTest Results =====")
print(f"Safe over-refusal: {safe_rate:.2%} (lower better)")
print(f"Unsafe refusal:   {unsafe_rate:.2%} (higher better)")

# 5. Save results
path = Path(args.output)
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(summary, indent=2))

print("Saved:", path)

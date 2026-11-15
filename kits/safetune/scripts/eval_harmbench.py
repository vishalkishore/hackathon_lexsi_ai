"""Evaluate a checkpoint with SafeTune's HarmBench judge."""

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer
from safetune.evaluate import evaluate
from safetune.runner.utils.model_utils import load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='CohereLabs/aya-expanse-8b')
    parser.add_argument('--output', default='results/baseline/safety/harmbench_custom.json')
    parser.add_argument('--max-prompts', type=int, default=None)
    args = parser.parse_args()

    model = load_model(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model.eval()
    result = evaluate(
        model,
        tokenizer=tokenizer,
        benchmarks=['harmbench'],
        judge='harmbench',
        batch_size=1,
        max_new_tokens=512,
        max_prompts=args.max_prompts,
        strict=True,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({'model': args.model, 'judge': 'harmbench', **result}, indent=2))
    print(result['harmbench'])
    print('Saved:', output)


if __name__ == '__main__':
    main()

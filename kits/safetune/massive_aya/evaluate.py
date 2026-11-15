#!/usr/bin/env python3
"""Greedy, paired evaluation of base Aya or a fresh SFT LoRA adapter."""
from __future__ import annotations

try:
    import _bootstrap
except ModuleNotFoundError:
    from aya_massive_en_hi import _bootstrap
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from massive_aya.core import (
    metrics, prompt_text, read_jsonl, read_manifest, score_record, write_jsonl, validate_pairs, SCORING_VERSION,
)

from massive_aya.provenance import MODEL_REVISION, environment, sha256, model_identity


def frozen_cast(model, tokenizer, cache_path, base_model, revision, active_policies, alpha):
    """Transfer a frozen base-Aya calibration without fitting on MASSIVE."""
    import torch
    from safetune.steer import CASTCondition
    from steering.policy_cast import PolicyCAST
    saved = torch.load(cache_path, map_location='cpu', weights_only=True)
    meta = saved['metadata']
    if (meta['version'] != 'binary-independent-policy-v1' or meta['model'] != base_model
            or meta['model_revision'] != revision or meta['dtype'] != 'bf16'
            or meta['chat_template'] != tokenizer.get_chat_template()):
        raise ValueError('Frozen CAST calibration identity/template/dtype mismatch')
    conditions = {name: CASTCondition(condition_vector=c['vector'], condition_layer=c['layer'],
        threshold=c['threshold'], comparator=c['comparator'], f1=c['fit_f1'], pool=c['pool'])
        for name, c in saved['conditions'].items()}
    controller = PolicyCAST(model, saved['vectors'], conditions, alpha)
    controller.set_active_policies(active_policies)
    return controller, {'cache_sha256': sha256(cache_path), 'calibration_metadata': meta,
                        'active_policies': active_policies, 'alpha': alpha,
                        'transfer': 'frozen_base_calibration_on_tuned_model',
                        'prompt_policy_injection': False,
                        'policy_cast_source_sha256': sha256(_bootstrap.ROOT / 'steering/policy_cast.py')}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("checkpoints/aya8b-massive-en-hi/prepared_data"))
    p.add_argument("--split", choices=["train", "validation", "test"], default="validation")
    p.add_argument("--allow-test", action="store_true", help="Explicit final-test permission")
    p.add_argument("--model", default="CohereLabs/aya-expanse-8b")
    p.add_argument("--model-revision", default=MODEL_REVISION)
    p.add_argument("--adapter", default=None, help="LoRA adapter folder; omit for original Aya")
    p.add_argument("--output", type=Path, required=True, help="Output directory, not an existing file")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--limit", type=int, default=0, help="Maximum rows; multiples of 2 keep bilingual pairs")
    p.add_argument("--max-new-tokens", type=int, default=192)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument('--cast-cache', type=Path, default=None)
    p.add_argument('--cast-policies', nargs='+', default=None)
    p.add_argument('--cast-alpha', type=float, default=0.3)
    args = p.parse_args()
    if bool(args.cast_cache) != bool(args.cast_policies):
        p.error('--cast-cache and --cast-policies must be supplied together')

    if args.split == "test" and not args.allow_test:
        p.error("Test is locked: set --allow-test only after selecting the checkpoint on validation")
    if args.batch_size < 1 or args.max_new_tokens < 1:
        p.error("Batch size and generation limit must be positive")
    if args.limit < 0 or (args.limit and args.limit % 2):
        p.error("--limit must be a positive even number, or 0 for entire split")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Output directory is nonempty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from peft import PeftModel

    if not torch.cuda.is_available():
        raise RuntimeError("This evaluation needs a CUDA GPU")
    set_seed(args.seed)

    manifest = read_manifest(args.data)
    all_rows = read_jsonl(args.data / f"{args.split}.jsonl")
    if args.limit:
        all_rows = all_rows[:args.limit]
    validate_pairs(all_rows)
    intents = manifest["intents"]
    slots = manifest["slot_types"]
    base_model, _ = model_identity(args.model, args.model_revision)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.model_revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    if not tokenizer.chat_template:
        raise ValueError("Model tokenizer has no chat template")
    if args.adapter:
        record = json.loads((Path(args.adapter).parent / "experiment.json").read_text())
        if record["model"] != base_model or record["model_revision"] != args.model_revision:
            raise ValueError("Adapter base identity differs from evaluation model")
        adapter_tokenizer = AutoTokenizer.from_pretrained(args.adapter)
        if adapter_tokenizer.get_chat_template() != tokenizer.get_chat_template():
            raise ValueError("Adapter and base chat templates differ")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.model_revision, dtype=torch.bfloat16, attn_implementation="sdpa",
        low_cpu_mem_usage=True, device_map={"": "cuda:0"},
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
    model.eval()
    controller = None
    cast_metadata = None
    if args.cast_cache:
        controller, cast_metadata = frozen_cast(model, tokenizer, args.cast_cache, base_model,
            args.model_revision, args.cast_policies, args.cast_alpha)

    run_metadata = {
        "environment": environment(), "model_revision": args.model_revision,
        "chat_template_sha256": hashlib.sha256(tokenizer.get_chat_template().encode()).hexdigest(),
        "adapter_sha256": sha256(Path(args.adapter) / "adapter_model.safetensors") if args.adapter else None,
        "cast": cast_metadata, "decoding": "greedy", "scoring_version": SCORING_VERSION, "partial": bool(args.limit), "batch_size": args.batch_size,
        "eos_token_id": tokenizer.eos_token_id, "pad_token_id": tokenizer.pad_token_id,
    }
    (args.output / "run_record.json").write_text(json.dumps(run_metadata, indent=2))
    all_out = []
    device = next(model.parameters()).device
    intents_set, slots_set = set(intents), set(slots)
    eos_tokens = [tokenizer.eos_token_id]
    end_turn = tokenizer.convert_tokens_to_ids("<|END_OF_TURN_TOKEN|>")
    if isinstance(end_turn, int) and end_turn != tokenizer.unk_token_id:
        eos_tokens.append(end_turn)
    eos_tokens = sorted(set(i for i in eos_tokens if i is not None))

    predictions = (args.output / "predictions.jsonl").open("x", encoding="utf-8")
    for start in range(0, len(all_rows), args.batch_size):
        group = all_rows[start:start+args.batch_size]
        prompts = [tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text(r["utt"], intents, slots)}],
            tokenize=False, add_generation_prompt=True,
        ) for r in group]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
        enc = {k: v.to(device) for k, v in enc.items()}
        if enc["input_ids"].shape[1] + args.max_new_tokens > model.config.max_position_embeddings:
            raise ValueError("Prompt and generation budget exceed model context")
        with torch.inference_mode():
            generations = (controller or model).generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                eos_token_id=eos_tokens,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )
        new_tokens = generations[:, enc["input_ids"].shape[1]:]
        texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        for index, (row, text) in enumerate(zip(group, texts)):
            scores = score_record(row, text, intents_set, slots_set)
            all_out.append({
                "id": row["id"], "locale": row["locale"],
                "intent": row["intent"], "utterance": row["utt"],
                "gold": row["answer"], "prediction": text,
                **scores,
                **({'cast_gate_fired': any(g['fires'][index] for g in controller.last_gate.values()),
                    'policy_gates': {name: {**g, 'fires': [g['fires'][index]],
                        'scores': [g['scores'][index]]} for name, g in controller.last_gate.items()}}
                   if controller else {}),
            })
        for result in all_out[-len(group):]:
            predictions.write(json.dumps(result, ensure_ascii=False) + "\n")
        predictions.flush()
        print(f"Evaluated {len(all_out)}/{len(all_rows)}", flush=True)

    output = args.output
    predictions.close()
    summary = {"overall": metrics(all_out)}
    for locale in ("en-US", "hi-IN"):
        subset = [x for x in all_out if x["locale"] == locale]
        summary[locale] = metrics(subset)
    per_intent = defaultdict(list)
    for r in all_out:
        per_intent[r["intent"]].append(r)
    if controller:
        summary['cast_gate_activation'] = {
            locale: {'n': len(subset), 'fired': sum(r['cast_gate_fired'] for r in subset)}
            for locale in ('overall', 'en-US', 'hi-IN')
            for subset in [[r for r in all_out if locale == 'overall' or r['locale'] == locale]]}
    summary["per_intent"] = {k: metrics(v) for k, v in sorted(per_intent.items())}
    summary["metadata"] = run_metadata | {
        "base_model": base_model, "evaluated_model": args.model, "adapter": args.adapter,
        "split": args.split, "seed": args.seed,
        "dataset_revision": manifest["source_revision"], "data_seed": manifest["seed"],
        "dataset_sha256": hashlib.sha256((args.data / f"{args.split}.jsonl").read_bytes()).hexdigest(),
        "n": len(all_out), "max_new_tokens": args.max_new_tokens,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_intent"}, indent=2))


if __name__ == "__main__":
    main()

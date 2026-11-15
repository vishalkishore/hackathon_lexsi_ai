#!/usr/bin/env python3
"""Policy-aware SafeTune CAST benchmark on requests.jsonl.

Important: SafeTune's default contrast-extraction path wraps each calibration
string as a USER turn. This script's calibration-tokenizer adapter preserves
already rendered SYSTEM-policy + conversation prompts, so calibration and
inference see the same token IDs. The adapter is not a new CAST algorithm.

Run `python cast_experiment.py inspect` without installing ML dependencies.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from typing import Any

HERE = Path(__file__).resolve().parent
DATA_DEFAULT = HERE / "requests.jsonl"
POLICY_DEFAULT = HERE / "policies.jsonl"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path: Path, *, binary_only: bool = True) -> list[dict]:
    rows = []
    seen = set()
    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            sid = row["sample_id"]
            if sid in seen:
                raise ValueError(f"Duplicate sample_id {sid} on line {n}")
            seen.add(sid)
            rows.append(row)
    allowed_actions = {'allow', 'refuse', 'partial', 'clarify'}
    if any(r['expected_action'] not in allowed_actions for r in rows):
        raise ValueError('Unknown expected_action in source dataset')
    families = defaultdict(set)
    for r in rows:
        families[r["family_id"]].add(r["split"])
    leaks = {k: v for k, v in families.items() if len(v) > 1}
    if leaks:
        raise ValueError(f"Family leakage across splits: {list(leaks.items())[:5]}")
    return [r for r in rows if r["expected_action"] in ("allow", "refuse")] if binary_only else rows


def inspect(args: argparse.Namespace) -> None:
    original = load_jsonl(args.data, binary_only=False)
    rows = [r for r in original if r["expected_action"] in ("allow", "refuse")]
    print("Excluded nonbinary rows:", len(original)-len(rows))
    print("Rows:", len(rows), "families:", len({r["family_id"] for r in rows}))
    print("Splits:", dict(Counter(r["split"] for r in rows)))
    print("Labels:", dict(Counter(r["expected_action"] for r in rows)))
    print("Languages:", dict(Counter(r["language"] for r in rows)))
    for sp in ("fit", "validation", "test", "stress_test"):
        sub = [r for r in rows if r["split"] == sp]
        print(sp, "n=", len(sub), "labels=", dict(Counter(r["expected_action"] for r in sub)))
    print("Policies:", dict(Counter("+".join(r["active_policies"]) for r in rows)))
    print("Source review:", dict(Counter(r.get("source", {}).get("review_status") for r in rows)))


def load_policies(path: Path) -> dict:
    """Use exact source policy texts; do not imply independent human review."""
    if path.suffix != '.jsonl':
        config = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(config, dict):
            raise ValueError('Policy JSON must be an object')
        return config
    from jsonschema import Draft202012Validator
    schema = json.loads((HERE / 'benchmark_v3/schemas/policy.schema.json').read_text())
    validator = Draft202012Validator(schema)
    specs = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        validator.validate(item)
        name = item['policy_id']
        if name in specs:
            raise ValueError(f'Duplicate policy specification: {name}')
        specs[name] = item
    if not specs:
        raise ValueError('Empty policy specifications')
    config = {name: item['policy_text'] for name, item in specs.items()}
    config['_specifications'] = specs
    config['_verified'] = False
    if all(item['core_safety_cannot_be_overridden'] for item in specs.values()):
        config['_shared_safety'] = 'The model’s underlying safety rules remain in force.'
    return config


def policy_text(r: dict, policies: dict) -> str:
    active = r["active_policies"]
    missing = [p for p in active if not isinstance(policies.get(p), str)]
    if missing:
        raise ValueError(f"Missing policy text for {missing}")
    # Exact text is held in this file, never copied into the user message.
    clauses = ["Active product policies (all apply; restrictions compose by union):"]
    if policies.get("_shared_safety"):
        clauses.insert(0, policies["_shared_safety"])
    clauses += [f"Policy {p}: {policies[p]}" for p in active]
    return "\n\n".join(clauses)


def messages_for(r: dict, policies: dict, include_policy: bool) -> list[dict]:
    msgs = []
    if include_policy:
        msgs.append({"role": "system", "content": policy_text(r, policies)})
    for msg in r.get("prior_messages", []):
        if msg["role"] not in ("user", "assistant"):
            raise ValueError(f"Unsupported prior message role in {r['sample_id']}")
        msgs.append({"role": msg["role"], "content": msg["content"]})
    msgs.append({"role": "user", "content": r["user_request"]})
    return msgs


def render(tokenizer, row: dict, policies: dict, policy: bool = True) -> str:
    return tokenizer.apply_chat_template(
        messages_for(row, policies, policy), tokenize=False, add_generation_prompt=True
    )


class CalibTokenizer:
    """Forward tokenizer calls; preserve pre-rendered, policy-aware prompts.

    SafeTune's _encode_prompts and CAA extractor reapply a one-user-turn chat
    template. Intercept exactly those calls for a known list of full prompts.
    All other tokenizer methods behave normally.
    """

    def __init__(self, inner, full_prompts: list[str]):
        object.__setattr__(self, "inner", inner)
        object.__setattr__(self, "full_prompts", set(full_prompts))

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def __setattr__(self, name, value):
        setattr(self.inner, name, value)

    def __call__(self, *args, **kwargs):
        return self.inner(*args, **kwargs)

    def apply_chat_template(self, conversation, *, tokenize=False, add_generation_prompt=True, **kwargs):
        if (len(conversation) == 1 and conversation[0].get("role") == "user"
                and conversation[0].get("content") in self.full_prompts):
            if tokenize:
                raise ValueError("Calibration wrapper expects tokenize=False")
            return conversation[0]["content"]
        return self.inner.apply_chat_template(
            conversation, tokenize=tokenize,
            add_generation_prompt=add_generation_prompt, **kwargs
        )


def choose_calibration(rows: list[dict], per_class: int, seed: int) -> tuple[list[dict], list[dict]]:
    """Balanced by (policy config, language) within allow/refuse; FIT only."""
    rng = random.Random(seed)
    result = {}
    for label in ("refuse", "allow"):
        buckets = defaultdict(list)
        for r in rows:
            if r["split"] == "fit" and r["expected_action"] == label:
                buckets[(tuple(r["active_policies"]), r["language"])].append(r)
        for a in buckets.values():
            rng.shuffle(a)
        ordered = sorted(buckets, key=str)
        selected = []
        i = 0
        while len(selected) < per_class:
            active = [k for k in ordered if i < len(buckets[k])]
            if not active:
                break
            rng.shuffle(active)
            for k in active:
                if len(selected) < per_class:
                    selected.append(buckets[k][i])
            i += 1
        result[label] = selected
    if not result["refuse"] or not result["allow"]:
        raise ValueError("Calibration requires FIT refuse and allow rows")
    return result["refuse"], result["allow"]


def policy_calibration(rows, per_class, seed):
    """Single-policy FIT contrasts provide unambiguous per-policy gate labels."""
    names = sorted({p for r in rows for p in r['active_policies']})
    result = {}
    for name in names:
        isolated = [r for r in rows if r['active_policies'] == [name]]
        refuse, allow = choose_calibration(isolated, per_class, seed)
        if len(refuse) != per_class or len(allow) != per_class:
            raise ValueError(f'Policy {name} needs {per_class} FIT rows per binary class')
        result[name] = (refuse, allow)
    return result


def select_eval(rows: list[dict], split: str, limit: int, seed: int,
                sample_ids: list[str] | None = None) -> list[dict]:
    chosen = [r for r in rows if r["split"] == split]
    if sample_ids is not None:
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError('Duplicate evaluation IDs')
        by_id = {r['sample_id']: r for r in chosen}
        missing = set(sample_ids) - set(by_id)
        if missing:
            raise ValueError(f'Evaluation IDs missing from requested split: {sorted(missing)}')
        return [by_id[sid] for sid in sample_ids]
    if not limit or limit >= len(chosen):
        return chosen
    # Select complete families where possible to retain policy-switch contrasts.
    families = defaultdict(list)
    for r in chosen:
        families[r["family_id"]].append(r)
    rng = random.Random(seed)
    groups = list(families.values())
    rng.shuffle(groups)
    selected = []
    for group in groups:
        if len(selected) + len(group) <= limit:
            selected.extend(group)
    if not selected:
        selected = sorted(chosen, key=lambda r: r["sample_id"])[:limit]
    return selected


def generation_batches(rows, modes, existing, batch_size):
    """Group by mode and active-policy tuple; never overlap steering hooks."""
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    for mode in modes:
        groups = defaultdict(list)
        for row in rows:
            if (row['sample_id'], mode) not in existing:
                groups[tuple(row['active_policies'])].append(row)
        for group in groups.values():
            for start in range(0, len(group), batch_size):
                yield mode, group[start:start + batch_size]


def gate_for_row(policy_gates, index):
    return {name: {**details, 'fires': [details['fires'][index]],
                  'scores': [details['scores'][index]]}
            for name, details in policy_gates.items()}


def run(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.batch_size < 1:
        raise ValueError('batch_size must be positive')
    rows = load_jsonl(args.data)
    policies = load_policies(args.policies)
    for r in rows:
        policy_text(r, policies)
    if args.split in ("test", "stress_test") and not policies.get("_verified", False):
        raise ValueError("Final evaluation requires verified policy text")
    if not policies.get("_verified", False):
        print("WARNING: Policies are not marked independently verified. Source JSONL texts are used exactly; benchmark human review is still pending.", file=sys.stderr)
    if args.split == "test" and not args.confirm_test:
        raise ValueError("Test is reserved for final evaluation. Pass --confirm-test to run it deliberately.")
    if args.split == "stress_test" and not args.confirm_test:
        raise ValueError("Stress test is held out. Pass --confirm-test to run it deliberately.")
    if args.split == "fit":
        raise ValueError("Do not evaluate on fit; use validation/test/stress_test")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.model_revision, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    kw = {"revision": args.model_revision, "dtype": torch.bfloat16 if args.dtype == "bf16" else torch.float16,
          "device_map": "auto", "low_cpu_mem_usage": True,
          "trust_remote_code": args.trust_remote_code}
    model = AutoModelForCausalLM.from_pretrained(args.model, **kw).eval()
    model.requires_grad_(False)
    device = next(model.parameters()).device

    refuse, allow = choose_calibration(rows, args.calib_per_class, args.seed)
    calibration_rows = refuse + allow
    by_policy = policy_calibration(rows, args.calib_per_class, args.seed)
    gate_rows = [r for refuse_rows, allow_rows in by_policy.values() for r in refuse_rows+allow_rows]
    rendered = [render(tokenizer, r, policies, True) for r in calibration_rows+gate_rows]
    for prompt in rendered:
        length = len(tokenizer(prompt, add_special_tokens=False)['input_ids'])
        if length > args.max_input_tokens or length > model.config.max_position_embeddings:
            raise ValueError('Calibration prompt exceeds input/context limit')
    print(f"Calibration: {len(refuse)} refuse and {len(allow)} allow; fitting only on split=fit")
    print("Policies in calibration:", dict(Counter("+".join(r["active_policies"]) for r in calibration_rows)))
    # Verify same model-facing token IDs whether passed directly or by adapter.
    adapter = CalibTokenizer(tokenizer, rendered)
    s = rendered[0]
    x = adapter.apply_chat_template([{"role": "user", "content": s}], tokenize=False)
    assert x == s
    assert tokenizer(s, add_special_tokens=False)["input_ids"] == adapter(x, add_special_tokens=False)["input_ids"]

    need_steering = any(m in args.modes for m in ("caa", "cast"))
    cast = caa = None
    if need_steering:
        from safetune.steer import (CAAModel, CASTCondition, CAAConfig,
                                    extract_caa_vectors, fit_cast_condition)
        from safetune._refusal_helpers import _get_decoder_layers
        from policy_cast import PolicyCAST
        cache_path = args.cache
        metadata = {"version": "binary-independent-policy-v1", "dataset_sha256": sha256(args.data),
                    "policies_sha256": sha256(args.policies), "model": args.model,
                    "model_revision": args.model_revision, "dtype": args.dtype,
                    "calib_per_class": args.calib_per_class, "seed": args.seed,
                    "condition_layers": args.condition_layers, "behavior_layers": args.behavior_layers,
                    "chat_template": tokenizer.get_chat_template(),
                    "safetune_version": __import__('importlib.metadata', fromlist=['version']).version('safetune'),
                    "sample_ids": [r["sample_id"] for r in calibration_rows],
                    "policy_sample_ids": {name: [r['sample_id'] for r in refuse_rows+allow_rows]
                                          for name, (refuse_rows, allow_rows) in by_policy.items()}}
        if cache_path.exists() and not args.recalibrate:
            saved = torch.load(cache_path, map_location="cpu", weights_only=True)
            if saved["metadata"] != metadata:
                raise ValueError("Cached calibration metadata differs. Use a new --cache or --recalibrate.")
            conditions = {name: CASTCondition(condition_vector=c['vector'], condition_layer=c['layer'],
                threshold=c['threshold'], comparator=c['comparator'], f1=c['fit_f1'], pool=c['pool'])
                for name, c in saved['conditions'].items()}
            if set(conditions) != set(by_policy):
                raise ValueError('Cached policy conditions do not match active policy inventory')
            vectors = saved['vectors']
        else:
            depth = len(_get_decoder_layers(model))
            behavior_layers = args.behavior_layers or list(range(round(14*depth/32), round(18*depth/32)+1))
            vectors = extract_caa_vectors(model, adapter,
                [render(tokenizer, r, policies) for r in refuse],
                [render(tokenizer, r, policies) for r in allow], CAAConfig(target_layers=behavior_layers))
            conditions = {}
            for name, (refuse_rows, allow_rows) in by_policy.items():
                conditions[name] = fit_cast_condition(model,
                    [render(tokenizer, r, policies) for r in refuse_rows],
                    [render(tokenizer, r, policies) for r in allow_rows], adapter,
                    candidate_layers=args.condition_layers, chat_template=True)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({'metadata': metadata,
                'conditions': {name: {'vector': c.condition_vector.cpu(), 'layer': int(c.condition_layer),
                    'threshold': float(c.threshold), 'comparator': c.comparator,
                    'fit_f1': float(c.f1), 'pool': c.pool} for name, c in conditions.items()},
                'vectors': {int(k): v.cpu() for k, v in vectors.items()}}, cache_path)
        cast = PolicyCAST(model, vectors, conditions, args.alpha)
        caa = CAAModel(model, vectors=vectors, strength=args.alpha)
        for name, condition in conditions.items():
            print(f'Policy {name}: condition layer={condition.condition_layer}, '
                  f'threshold={condition.threshold:.4f}, comparator={condition.comparator}, '
                  f'internal_fit_F1={condition.f1:.4f}')

    output_rows = select_eval(rows, args.split, args.limit, args.seed,
                              getattr(args, 'sample_ids', None))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    existing = set()
    if args.output.exists() and not args.overwrite:
        with args.output.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    existing.add((r["sample_id"], r["mode"]))
    if args.overwrite:
        args.output.unlink(missing_ok=True)

    settings = {"model": args.model, "model_revision": args.model_revision,
                "experiment_version": "binary-independent-policy-v1",
                "condition_layers": args.condition_layers, "behavior_layers": args.behavior_layers,
                "batch_size": args.batch_size, "max_input_tokens": args.max_input_tokens, "dtype": args.dtype, "limit": args.limit, "dataset_sha256": sha256(args.data),
                "policies_sha256": sha256(args.policies), "policy_verified": policies.get("_verified"),
                "split": args.split, "alpha": args.alpha, "seed": args.seed,
                "max_new_tokens": args.max_new_tokens, "calib_per_class": args.calib_per_class,
                "modes": args.modes, "cache": str(args.cache),
                "transformers_version": __import__("transformers").__version__,
                "torch_version": torch.__version__, "safetune_version": getattr(__import__("safetune"), "__version__", "unknown")}
    settings_path = args.output.with_suffix(".settings.json")
    if getattr(args, 'sample_ids', None) is not None:
        settings['sample_ids'] = args.sample_ids
    if settings_path.exists() and existing:
        old = json.loads(settings_path.read_text(encoding="utf-8"))
        if old != settings:
            raise ValueError("Output exists with different settings. Use a new --output or --overwrite.")
    settings_path.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    total = len(output_rows)*len(args.modes)
    completed = sum((r['sample_id'], mode) in existing
                    for r in output_rows for mode in args.modes)
    with args.output.open("a", encoding="utf-8") as out:
        for mode, batch_rows in generation_batches(output_rows, args.modes, existing, args.batch_size):
            prompts = [render(tokenizer, row, policies, mode != 'base') for row in batch_rows]
            batch = tokenizer(prompts, padding=True, return_tensors="pt",
                              add_special_tokens=False).to(device)
            width = batch['input_ids'].shape[-1]
            if width > args.max_input_tokens:
                raise ValueError("Prompt exceeds input limit")
            if width + args.max_new_tokens > model.config.max_position_embeddings:
                raise ValueError("Prompt plus generation exceeds model context")
            gate = {}
            if mode == "cast":
                cast.set_active_policies(batch_rows[0]["active_policies"])
                prev = cast._gate_fires
                def instrument(*a, **k):
                    fires, scores = prev(*a, **k)
                    gate.update({"fires": fires, "scores": scores})
                    return fires, scores
                cast._gate_fires = instrument
            start = time.perf_counter()
            try:
                target = {"base": model, "policy": model, "caa": caa, "cast": cast}[mode]
                with torch.inference_mode():
                    output = target.generate(
                        **batch, max_new_tokens=args.max_new_tokens,
                        do_sample=False, use_cache=True,
                        pad_token_id=tokenizer.pad_token_id,
                    )
            finally:
                if mode == "cast":
                    cast._gate_fires = prev
            elapsed = time.perf_counter() - start
            for index, row in enumerate(batch_rows):
                new = output[index, width:].tolist()
                # Generation pads finished rows until the longest row finishes.
                eos = model.generation_config.eos_token_id
                eos_ids = set(eos if isinstance(eos, list) else [eos])
                stop = next((i + 1 for i, token in enumerate(new) if token in eos_ids), len(new))
                new = new[:stop]
                text = tokenizer.decode(new, skip_special_tokens=True).strip()
                fired = gate['fires'][index] if mode == 'cast' else None
                score = gate['scores'][index] if mode == 'cast' else None
                record = {
                    "sample_id": row["sample_id"], "family_id": row["family_id"],
                    "split": row["split"], "mode": mode,
                    "active_policies": row["active_policies"],
                    "language": row["language"], "expected_action": row["expected_action"],
                    "core_safety_label": row["core_safety_label"],
                    "challenge_tags": row["challenge_tags"],
                    "gate_fired": fired, "gate_score": score,
                    "policy_gates": gate_for_row(cast.last_gate, index) if mode == "cast" else None,
                    "generated_text": text, "num_new_tokens": len(new),
                    "elapsed_sec": round(elapsed / len(batch_rows), 4),
                    "batch_elapsed_sec": round(elapsed, 4), "batch_size": len(batch_rows),
                }
                out.write(json.dumps(record, ensure_ascii=False)+"\n")
                out.flush()
                completed += 1
                print(f"[{completed}/{total}] {row['sample_id']} {mode:6s} "
                      f"expected={row['expected_action']:7s} gate={fired} "
                      f"batch={len(batch_rows)} {elapsed:.1f}s/batch", flush=True)
    print("Saved", args.output)


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("inspect", help="Inspect data without GPU")
    p.add_argument("--data", type=Path, default=DATA_DEFAULT)
    p.set_defaults(func=inspect)

    p = sub.add_parser("run", help="Generate outputs with base/policy/CAA/CAST")
    p.add_argument("--data", type=Path, default=DATA_DEFAULT)
    p.add_argument("--policies", type=Path, default=POLICY_DEFAULT)
    p.add_argument("--model-revision", default="5062468bf9bc0c6035fd64e06274333ec127d980")
    p.add_argument("--model", default="CohereLabs/aya-expanse-8b")
    p.add_argument("--output", type=Path, default=HERE / "runs" / "validation.jsonl")
    p.add_argument("--cache", type=Path, default=HERE / "runs" / "cast_calibration.pt")
    p.add_argument("--split", choices=["validation", "test", "stress_test"], default="validation")
    p.add_argument("--modes", nargs="+", choices=["base", "policy", "caa", "cast"],
                   default=["base", "policy", "caa", "cast"])
    p.add_argument("--batch-size", type=int, default=4, help="Prompts per generation batch; use 1 for serial execution")
    p.add_argument("--limit", type=int, default=20, help="0 means full split")
    p.add_argument("--sample-ids", nargs='+', default=None,
                   help="Exact evaluation IDs in requested order; overrides --limit")
    p.add_argument("--calib-per-class", type=int, default=40)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--condition-layers", type=int, nargs="+", default=None)
    p.add_argument("--behavior-layers", type=int, nargs="+", default=None)
    p.add_argument("--max-new-tokens", type=int, default=1800)
    p.add_argument("--max-input-tokens", type=int, default=4096)
    p.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--confirm-test", action="store_true")
    p.add_argument("--recalibrate", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--trust-remote-code", action="store_true")
    p.set_defaults(func=run)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    cli()

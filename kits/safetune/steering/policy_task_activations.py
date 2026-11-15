"""Unsteered final-prompt-token activations and matched policy/task contrasts."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_prompts(path):
    """Read model inputs only; never open references or behavior suffixes."""
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    groups = defaultdict(dict)
    ids = set()
    families = defaultdict(set)
    for row in rows:
        pid, condition = row['problem_id'], row['condition']
        if row['id'] in ids or condition in groups[pid] or condition not in 'ABCD':
            raise ValueError('Duplicate ID or invalid condition')
        ids.add(row['id'])
        if row['messages'] != [{'role': 'system', 'content': row['system']},
                               {'role': 'user', 'content': row['user']}]:
            raise ValueError('Expected trusted system and unchanged user messages')
        if any(k in row for k in ('reference_python', 'reference_explanation',
                                 'positive_explanation', 'negative_python')):
            raise ValueError('Reference content must remain separate')
        groups[pid][condition] = row
        families[row['family']].add(row['split_family_ood'])
    if not rows or any(len(splits) != 1 for splits in families.values()):
        raise ValueError('Empty data or family split leakage')
    for quartet in groups.values():
        if set(quartet) != set('ABCD'):
            raise ValueError('Incomplete quartet')
        a, b, c, d = (quartet[k] for k in 'ABCD')
        if a['user'] != b['user'] or c['user'] != d['user']:
            raise ValueError('User request changes across policy pair')
        if a['system'] != c['system'] or b['system'] != d['system'] or a['system'] == b['system']:
            raise ValueError('System policy mismatch')
        for key in ('family', 'split_family_ood', 'split_stratified'):
            if len({r[key] for r in quartet.values()}) != 1:
                raise ValueError(f'Quartet {key} mismatch')
        for condition, row in quartet.items():
            if row['policy_state'] != ('on' if condition in 'BD' else 'off'):
                raise ValueError('Incorrect policy state')
            if row['request_type'] != ('code' if condition in 'AB' else 'explanation'):
                raise ValueError('Incorrect request type')
    return rows


def contrasts(activations, rows):
    """Rows × layers × width -> problems × layers × width, preserving pairing."""
    import torch
    if activations.ndim != 3 or activations.shape[0] != len(rows):
        raise ValueError('Activation shape mismatch')
    groups = defaultdict(dict)
    for i, row in enumerate(rows):
        if row['condition'] in groups[row['problem_id']]:
            raise ValueError('Duplicate condition')
        groups[row['problem_id']][row['condition']] = i
    result = []
    for pid, quartet in groups.items():
        if set(quartet) != set('ABCD'):
            raise ValueError('Incomplete quartet')
        a, b, c, d = (activations[quartet[k]].float() for k in 'ABCD')
        result.append((pid, b - a, d - c))
    code = torch.stack([r[1] for r in result])
    explanation = torch.stack([r[2] for r in result])
    return {'problem_ids': [r[0] for r in result], 'policy_effect_code': code,
            'policy_effect_explanation': explanation, 'interaction': code - explanation}


def extract_final(model, tokenizer, rows, layers, max_tokens):
    """Single unpadded prefills; block outputs before final model normalization."""
    import torch
    decoder = model.model.layers
    if not layers or len(set(layers)) != len(layers) or any(i < 0 or i >= len(decoder) for i in layers):
        raise ValueError('Invalid decoder layers')
    captured = {}
    def hook(layer):
        def capture(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            captured[layer] = h[0, -1].detach().float().cpu().clone()
        return capture
    handles = [decoder[layer].register_forward_hook(hook(layer)) for layer in layers]
    values, tokens = [], []
    try:
        for i, row in enumerate(rows):
            enc = tokenizer.apply_chat_template(row['messages'], tokenize=True,
                add_generation_prompt=True, return_tensors='pt', return_dict=True)
            ids = enc['input_ids'][0].tolist()
            if not ids or len(ids) > max_tokens:
                raise ValueError(f"Invalid prompt length: {row['id']}: {len(ids)}")
            captured.clear()
            with torch.inference_mode():
                model(**enc.to(model.device), use_cache=False)
            values.append(torch.stack([captured[layer] for layer in layers]))
            tokens.append({'id': row['id'], 'token_count': len(ids), 'final_token_id': ids[-1],
                           'final_token_text': tokenizer.decode([ids[-1]]), 'input_ids': ids})
            print(f"Prefill {i+1}/{len(rows)} {row['id']} tokens={len(ids)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
    activations = torch.stack(values)
    if not torch.isfinite(activations).all():
        raise ValueError('Nonfinite activations')
    return activations, tokens


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--model', default='CohereLabs/aya-expanse-8b')
    parser.add_argument('--revision', required=True)
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--layers', type=int, nargs='+')
    parser.add_argument('--max-problems', type=int)
    parser.add_argument('--max-input-tokens', type=int, default=8192)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    rows = load_prompts(args.data)
    if args.max_problems is not None:
        if args.max_problems < 1:
            parser.error('--max-problems must be positive')
        selected = list(dict.fromkeys(r['problem_id'] for r in rows))[:args.max_problems]
        rows = [r for r in rows if r['problem_id'] in selected]
    args.out.mkdir(parents=True, exist_ok=False)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    set_seed(args.seed)
    started = time.monotonic()
    manifest = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True)
    manifest.update(status='running', started_utc=datetime.now(timezone.utc).isoformat(),
        git_commit=commit.stdout.strip() if commit.returncode == 0 else None,
        dataset_sha256=digest(args.data), source_sha256=digest(__file__), uv_lock_sha256=digest('uv.lock'),
        versions={p: version(p) for p in ('torch', 'transformers', 'accelerate')},
        environment={k: os.environ.get(k) for k in
                     ('CUDA_VISIBLE_DEVICES', 'HF_DEACTIVATE_ASYNC_LOAD', 'OMP_NUM_THREADS')},
        python=__import__('sys').version, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(torch.device(args.device)),
        activation_site='decoder block output, before final model normalization',
        token_site='last unpadded chat-template token with add_generation_prompt=True',
        formula='(B - A) - (D - C)', normalization='none; BF16 forward, FP32 saved and subtracted',
        split_scheme='split_family_ood', count=len(rows), batch_size=1, steering='none',
        generation='none; prefill only', local_files_only=True)
    def save_manifest():
        (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    save_manifest()
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision,
            dtype=torch.bfloat16, device_map={'': args.device}, local_files_only=True).eval()
        model.requires_grad_(False)
        layers = args.layers if args.layers is not None else list(range(len(model.model.layers)))
        manifest.update(layers=layers, chat_template=tokenizer.get_chat_template(),
                        resolved_revision=getattr(model.config, '_commit_hash', None))
        save_manifest()
        h, tokens = extract_final(model, tokenizer, rows, layers, args.max_input_tokens)
        paired = contrasts(h, rows)
        torch.save({'sample_ids': [r['id'] for r in rows], 'layers': layers, 'activations': h},
                   args.out / 'activations.pt')
        torch.save({**paired, 'layers': layers}, args.out / 'contrasts.pt')
        (args.out / 'tokens.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in tokens))
        (args.out / 'requests.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
        summary = []
        for i, pid in enumerate(paired['problem_ids']):
            row = next(r for r in rows if r['problem_id'] == pid)
            summary.append({'problem_id': pid, 'family': row['family'],
                'split_family_ood': row['split_family_ood'], 'layer_l2': {
                    key: paired[key][i].norm(dim=-1).tolist() for key in
                    ('policy_effect_code', 'policy_effect_explanation', 'interaction')}})
        (args.out / 'per_problem.json').write_text(json.dumps(summary, indent=2) + '\n')
        manifest.update(status='complete', activation_shape=list(h.shape),
                        elapsed_seconds=time.monotonic() - started,
                        completed_utc=datetime.now(timezone.utc).isoformat())
    except Exception as exc:
        manifest.update(status='failed', error=repr(exc), elapsed_seconds=time.monotonic() - started)
        raise
    finally:
        save_manifest()


if __name__ == '__main__':
    main()

"""Fit and run IBM CAST unchanged, with policy-aware formatting and split isolation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from original_cast_data import read, sha, write_json, write_rows

ROOT = Path(__file__).resolve().parents[1]
LIBRARY = ROOT / 'activation-steering'
sys.path.insert(0, str(LIBRARY))


def gate_decision(score, threshold, comparator):
    # These names are reversed in this pinned IBM implementation. Preserve them.
    if comparator == 'smaller':
        return score > threshold
    if comparator == 'larger':
        return score < threshold
    raise ValueError('Unknown IBM comparator')


def choose_gate(scores, layers):
    candidates = []
    positives = sum(row['target'] for row in scores)
    if not 0 < positives < len(scores):
        raise ValueError('Gate calibration needs both classes')
    for layer in layers:
        for tick in range(-200, 201):
            threshold = tick / 200
            for comparator in ('smaller', 'larger'):
                predictions = [gate_decision(row['scores'][str(layer)], threshold, comparator) for row in scores]
                tp = sum(p and r['target'] for p, r in zip(predictions, scores))
                fp = sum(p and not r['target'] for p, r in zip(predictions, scores))
                fn = positives - tp
                accuracy = sum(p == r['target'] for p, r in zip(predictions, scores)) / len(scores)
                f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
                candidates.append((f1, accuracy, -layer, -threshold, comparator))
    f1, accuracy, negative_layer, negative_threshold, comparator = max(candidates)
    return {'layer': -negative_layer, 'threshold': -negative_threshold, 'comparator': comparator,
            'dev_f1': f1, 'dev_accuracy': accuracy, 'dev_n': len(scores), 'dev_positive': positives,
            'score_site': 'original LeashLayer block input; final prompt token; tanh projection cosine'}


class RenderedTokenizer:
    """Original vector extractor receives fully templated strings without a second BOS."""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

    def __call__(self, *args, **kwargs):
        kwargs['add_special_tokens'] = False
        return self.tokenizer(*args, **kwargs)

    @property
    def pad_token_id(self):
        return self.tokenizer.pad_token_id

    @pad_token_id.setter
    def pad_token_id(self, value):
        self.tokenizer.pad_token_id = value


def original_dataset(tokenizer, pairs):
    """Use IBM formatting per pair, preserving trusted roles and paired response suffixes."""
    from activation_steering import SteeringDataset
    combined = None
    for positive, negative, positive_suffix, negative_suffix in pairs:
        if len(positive) != 2 or len(negative) != 2:
            raise ValueError('Expected system/user messages')
        positive_full = tokenizer.apply_chat_template(positive, tokenize=False, add_generation_prompt=True)
        negative_full = tokenizer.apply_chat_template(negative, tokenize=False, add_generation_prompt=True)
        header = positive_full[len(tokenizer.apply_chat_template(positive, tokenize=False,
                                                                 add_generation_prompt=False)):]
        dataset = SteeringDataset(tokenizer, examples=[(positive[1]['content'], negative[1]['content'])],
            system_message=(positive[0]['content'], negative[0]['content']),
            suffixes=[(header + positive_suffix, header + negative_suffix)])
        actual = dataset.formatted_dataset[0]
        if actual.positive != positive_full + positive_suffix or actual.negative != negative_full + negative_suffix:
            raise ValueError('IBM formatting changed the trusted prompt')
        if combined is None:
            combined = dataset
        else:
            combined.formatted_dataset.extend(dataset.formatted_dataset)
    if combined is None:
        raise ValueError('No training pairs')
    return combined


def load_model(config, device):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from activation_steering.config import GlobalConfig
    GlobalConfig.set_verbose(False)
    set_seed(config['seed'])
    tokenizer = AutoTokenizer.from_pretrained(config['model'], revision=config['revision'], local_files_only=True)
    # Original IBM extraction calls Tensor.numpy(); FP16 supports it, BF16 does not.
    model = AutoModelForCausalLM.from_pretrained(config['model'], revision=config['revision'],
        dtype=torch.float16, device_map={'': device}, local_files_only=True).eval()
    model.requires_grad_(False)
    return model, tokenizer


def environment(config):
    import torch
    commit = subprocess.run(['git', '-C', str(LIBRARY), 'rev-parse', 'HEAD'], capture_output=True, text=True)
    return {'utc': datetime.now(timezone.utc).isoformat(), 'python': platform.python_version(),
        'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(),
        'versions': {k: version(k) for k in ('torch', 'transformers', 'scikit-learn', 'numpy')},
        'original_cast_commit': commit.stdout.strip(),
        'original_cast_source_hashes': {p.name: sha(p) for p in (LIBRARY / 'activation_steering').glob('*.py')},
        'runner_sha256': sha(__file__), 'data_source_sha256': sha(Path(__file__).with_name('original_cast_data.py')),
        'uv_lock_sha256': sha(ROOT / 'uv.lock'),
        'environment': {k: os.environ.get(k) for k in
                        ('CUDA_VISIBLE_DEVICES', 'HF_DEACTIVATE_ASYNC_LOAD', 'OMP_NUM_THREADS')},
        'note': 'Original IBM vectors, tanh projection gate and block-input injection; no SafeTune'}


def load_config(data):
    config = json.loads((data / 'config.json').read_text())
    for name, expected in config['input_sha256'].items():
        if sha(data / name) != expected:
            raise ValueError('Prepared data hash mismatch')
    return config


def fit(args):
    import torch
    from activation_steering import MalleableModel, SteeringVector
    from activation_steering.leash_layer import LeashLayer
    config = load_config(args.data)
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / 'config.json', config)
    write_json(args.out / 'environment.json', environment(config))
    model, tokenizer = load_model(config, args.device)
    write_json(args.out / 'chat_template.json', {'chat_template': tokenizer.get_chat_template()})
    adapter = RenderedTokenizer(tokenizer)
    behaviors = read(args.data / 'behavior_train.jsonl')
    conditions = read(args.data / 'condition_train.jsonl')
    if any(r['split'] != 'train' for r in behaviors + conditions):
        raise ValueError('Non-training data in extraction')
    started = time.monotonic()
    for name in ('refusal', 'explanation', 'condition'):
        if name == 'condition':
            pairs = [(r['positive_messages'], r['negative_messages'], '', '') for r in conditions]
            layers = config['condition_layers']
        else:
            pairs = [(r['messages'], r['messages'], r[name], r['implementation']) for r in behaviors]
            layers = config['behavior_layers']
        dataset = original_dataset(tokenizer, pairs)
        vector = SteeringVector.train(model, adapter, dataset, hidden_layer_ids=layers,
            batch_size=1, method=config['vector_method'], accumulate_last_x_tokens=1)
        vector.save(str(args.out / name))
        print(f'Saved original IBM {name} vector from {len(pairs)} TRAIN pairs', flush=True)
    wrapper = MalleableModel(model, tokenizer)
    condition = SteeringVector.load(str(args.out / 'condition'))
    scores = []
    dev = [r for r in read(args.data / 'requests.jsonl') if r['split_family_ood'] == 'dev']
    try:
        for i, row in enumerate(dev):
            wrapper.reset_leash_to_default()
            wrapper.steer(condition_vector=condition, condition_layer_ids=config['condition_layers'],
                condition_vector_threshold=2.0, condition_comparator_threshold_is='smaller',
                condition_threshold_comparison_mode='last')
            enc = tokenizer.apply_chat_template(row['messages'], tokenize=True,
                add_generation_prompt=True, return_dict=True, return_tensors='pt').to(model.device)
            with torch.inference_mode():
                model(**enc, use_cache=False)
            score = {str(layer): LeashLayer.condition_similarities[0][layer] for layer in config['condition_layers']}
            if not all(__import__('math').isfinite(v) for v in score.values()):
                raise ValueError('Nonfinite gate score')
            scores.append({'id': row['id'], 'problem_id': row['problem_id'],
                           'target': row['policy_conflict'], 'scores': score})
            print(f'Gate DEV prefill {i+1}/{len(dev)}', flush=True)
    finally:
        wrapper.reset_leash_to_default()
        wrapper.unwrap()
    write_rows(args.out / 'gate_dev.jsonl', scores)
    gate = choose_gate(scores, config['condition_layers'])
    write_json(args.out / 'gate.json', gate)
    write_json(args.out / 'manifest.json', {'status': 'complete', 'train_problem_ids': [r['problem_id'] for r in behaviors],
        'train_pairs': {'refusal': len(behaviors), 'explanation': len(behaviors), 'condition': len(conditions)},
        'elapsed_seconds': time.monotonic() - started,
        'vector_sha256': {name: sha(args.out / (name + '.svec')) for name in ('refusal', 'explanation', 'condition')},
        'gate_sha256': sha(args.out / 'gate.json')})


def choose_rows(data, phase):
    rows = read(data / 'requests.jsonl')
    if phase == 'smoke':
        return [r for r in rows if r['problem_id'] == 'ALG001' and r['condition'] in 'ABCD'] + [
            r for r in rows if r['problem_id'] == 'ALG002' and r['condition'] == 'M']
    return [r for r in rows if r['split_family_ood'] == phase]


def arms_for(config, phase, selection):
    if phase == 'smoke':
        strengths = {'refusal': [1.5], 'explanation': [1.5]}
    elif phase == 'dev':
        strengths = {k: config['strengths'] for k in ('refusal', 'explanation')}
    else:
        if not selection or selection.get('selected_on') != 'dev':
            raise ValueError('Test requires a frozen DEV selection')
        strengths = {k: [selection['strengths'][k]] for k in ('refusal', 'explanation')}
        if any(v[0] not in config['strengths'] for v in strengths.values()):
            raise ValueError('Selected strength was not in the dev grid')
    return [('prompt', 0.0)] + [(f'cast_{k}', strength) for k, values in strengths.items() for strength in values]


def generate(args):
    import torch
    from activation_steering import MalleableModel, SteeringVector
    from activation_steering.leash_layer import LeashLayer
    config = load_config(args.data)
    if json.loads((args.fit / 'config.json').read_text()) != config:
        raise ValueError('Fit configuration mismatch')
    fitted = json.loads((args.fit / 'manifest.json').read_text())
    for name, expected in fitted['vector_sha256'].items():
        if sha(args.fit / (name + '.svec')) != expected:
            raise ValueError('Vector hash mismatch')
    if sha(args.fit / 'gate.json') != fitted['gate_sha256']:
        raise ValueError('Gate hash mismatch')
    rows = choose_rows(args.data, args.phase)
    selection = json.loads(args.selection.read_text()) if args.selection else None
    arms = arms_for(config, args.phase, selection)
    if args.phase == 'test' and selection['fit_manifest_sha256'] != sha(args.fit / 'manifest.json'):
        raise ValueError('Selection came from a different fit')
    settings = {'phase': args.phase, 'config': config, 'fit_manifest_sha256': sha(args.fit / 'manifest.json'),
        'arms': arms, 'sample_ids': [r['id'] for r in rows], 'seed': config['seed'],
        'max_new_tokens': config['smoke_max_new_tokens'] if args.phase == 'smoke' else config['max_new_tokens'],
        'selection_sha256': sha(args.selection) if args.selection else None}
    if args.resume:
        if json.loads((args.out / 'settings.json').read_text()) != json.loads(json.dumps(settings)):
            raise ValueError('Resume settings mismatch')
    else:
        args.out.mkdir(parents=True, exist_ok=False)
        write_json(args.out / 'settings.json', settings)
        write_rows(args.out / 'requests.jsonl', rows)
    write_json(args.out / 'environment.json', environment(config))
    response_path = args.out / 'responses.jsonl'
    existing = read(response_path) if response_path.exists() else []
    done = {(r['id'], r['method'], r['strength']) for r in existing}
    if len(done) != len(existing):
        raise ValueError('Duplicate saved response')
    model, tokenizer = load_model(config, args.device)
    write_json(args.out / 'chat_template.json', {'chat_template': tokenizer.get_chat_template()})
    gate = json.loads((args.fit / 'gate.json').read_text())
    vectors = {k: SteeringVector.load(str(args.fit / k)) for k in ('refusal', 'explanation', 'condition')}
    wrapper = MalleableModel(model, tokenizer)
    started = time.monotonic()
    try:
        with response_path.open('a') as output:
            for method, strength in arms:
                for row in rows:
                    if (row['id'], method, strength) in done:
                        continue
                    wrapper.reset_leash_to_default()
                    if method != 'prompt':
                        wrapper.steer(behavior_vector=vectors[method.removeprefix('cast_')],
                            behavior_layer_ids=config['behavior_layers'], behavior_vector_strength=strength,
                            condition_vector=vectors['condition'], condition_layer_ids=[gate['layer']],
                            condition_vector_threshold=gate['threshold'],
                            condition_comparator_threshold_is=gate['comparator'],
                            condition_threshold_comparison_mode='last', apply_behavior_on_first_call=True,
                            use_ooi_preventive_normalization=False)
                    enc = tokenizer.apply_chat_template(row['messages'], tokenize=True,
                        add_generation_prompt=True, return_dict=True, return_tensors='pt').to(model.device)
                    width = enc['input_ids'].shape[1]
                    if width + settings['max_new_tokens'] > model.config.max_position_embeddings:
                        raise ValueError('Context overflow; truncation forbidden')
                    tick = time.monotonic()
                    with torch.inference_mode():
                        generated = wrapper.generate(**enc, max_new_tokens=settings['max_new_tokens'],
                            do_sample=False, repetition_penalty=1.0, use_cache=True,
                            pad_token_id=tokenizer.eos_token_id)
                    tokens = generated[0, width:].tolist()
                    text = tokenizer.decode(tokens, skip_special_tokens=True).strip()
                    eos = model.generation_config.eos_token_id
                    eos_ids = eos if isinstance(eos, list) else [eos]
                    record = {**row, 'method': method, 'strength': strength, 'generated_text': text,
                        'output_tokens': tokens, 'input_token_ids': enc['input_ids'][0].tolist(),
                        'elapsed_seconds': time.monotonic() - tick,
                        'truncated': len(tokens) == settings['max_new_tokens'] and tokens[-1] not in eos_ids,
                        'gate_fired': bool(LeashLayer.condition_met[0]) if method != 'prompt' else None,
                        'gate_score': LeashLayer.condition_similarities[0].get(gate['layer']) if method != 'prompt' else None}
                    output.write(json.dumps(record, ensure_ascii=False) + '\n')
                    output.flush()
                    done.add((row['id'], method, strength))
                    print(f'{len(done)}/{len(rows)*len(arms)} {method} alpha={strength} {row["id"]} '
                          f'tokens={len(tokens)} gate={record["gate_fired"]}', flush=True)
    finally:
        wrapper.reset_leash_to_default()
        wrapper.unwrap()
        write_json(args.out / 'manifest.json', {'status': 'complete' if len(done) == len(rows)*len(arms) else 'incomplete',
            'count': len(done), 'expected': len(rows)*len(arms), 'elapsed_seconds': time.monotonic() - started})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('fit', 'generate'):
        p = sub.add_parser(name)
        p.add_argument('--data', type=Path, required=True)
        p.add_argument('--out', type=Path, required=True)
        p.add_argument('--device', default='cuda:0')
        if name == 'generate':
            p.add_argument('--fit', type=Path, required=True)
            p.add_argument('--phase', choices=['smoke', 'dev', 'test'], required=True)
            p.add_argument('--selection', type=Path)
            p.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.command == 'fit':
        fit(args)
    else:
        generate(args)

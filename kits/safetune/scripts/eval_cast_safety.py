#!/usr/bin/env python3
"""Safety evaluation through frozen CAST, using existing benchmark loaders/judges."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from aya_massive_en_hi.evaluate import frozen_cast
from aya_massive_en_hi.src.massive_aya.provenance import MODEL_REVISION, environment, sha256

POLICIES = ['coding_tutor_v1', 'legal_information_v1', 'general_assistant_v1']
BASE = 'CohereLabs/aya-expanse-8b'


class AuditedCAST:
    def __init__(self, controller, model, tokenizer, path):
        self.controller, self.model, self.tokenizer = controller, model, tokenizer
        self.device = next(model.parameters()).device
        self.path = path
        self.subset = None

    def generate(self, **kwargs):
        ids = kwargs['input_ids']
        if ids.shape[1] + kwargs['max_new_tokens'] > self.model.config.max_position_embeddings:
            raise ValueError('Safety prompt plus generation exceeds context; refusing truncation')
        output = self.controller.generate(**kwargs)
        prompts = self.tokenizer.batch_decode(ids, skip_special_tokens=False)
        responses = self.tokenizer.batch_decode(output[:, ids.shape[1]:], skip_special_tokens=True)
        with self.path.open('a', encoding='utf-8') as f:
            for i, (prompt, response) in enumerate(zip(prompts, responses)):
                gates = {name: {**g, 'fires': [g['fires'][i]], 'scores': [g['scores'][i]]}
                         for name, g in self.controller.last_gate.items()}
                f.write(json.dumps({'subset': self.subset, 'rendered_prompt': prompt,
                    'response': response, 'policy_gates': gates,
                    'gate_fired': any(g['fires'][0] for g in gates.values())}, ensure_ascii=False)+'\n')
        return output


def restore_polyguard(rows, records, tokenizer, batch_size):
    """Recover flushed prefix responses only after exact rendered-prompt checks."""
    if len(records) > len(rows):
        raise ValueError('Audit has more responses than the input manifest')
    if len(records) != len(rows) and len(records) % batch_size:
        raise ValueError('Audit ends inside a batch; preserve it for explicit partial-batch recovery')
    for start in range(0, len(records), batch_size):
        group = rows[start:start+batch_size]
        prompts = [tokenizer.apply_chat_template([{'role':'user','content':r['prompt']}],
            tokenize=False, add_generation_prompt=True) for r in group]
        enc = tokenizer(prompts, padding=True, return_tensors='pt', add_special_tokens=False)
        rendered = tokenizer.batch_decode(enc['input_ids'], skip_special_tokens=False)
        for offset, (row, expected) in enumerate(zip(group, rendered)):
            record = records[start+offset]
            if record['rendered_prompt'] != expected:
                raise ValueError(f'Audit prompt mismatch at row {start+offset}')
            if ('id' in record and (record['id'], record['language']) != (row['id'], row['language'])):
                raise ValueError('Audit ID/language mismatch')
            row['response'] = record['response']
            row['input_tokens'] = int(enc['attention_mask'][offset].sum())
    return len(records)


def check_resume_identity(old, new):
    for field in ('adapter_sha256','cache_sha256','alpha','active_policies','task','batch_size',
                  'max_new_tokens','seed','decoding','limit_per_language','max_prompts'):
        if old[field] != new[field]:
            raise ValueError(f'Resume setting differs: {field}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--resume', action='store_true', help='Resume PolyGuard from verified audit prefix')
    p.add_argument('--task', required=True, choices=['polyguard', 'xstest_split', 'harmbench_custom',
        'advbench', 'wildjailbreak', 'orbench_hard', 'xstest', 'harmbench'])
    p.add_argument('--adapter', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--max-prompts', type=int, default=None, help='Diagnostic cap; omitted for full benchmark')
    p.add_argument('--limit-per-language', type=int, default=0)
    args = p.parse_args()
    if args.resume and args.task != 'polyguard':
        p.error('--resume is supported only for PolyGuard generation')
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        p.error('Output is nonempty; use a fresh directory or --resume for PolyGuard')
    previous = None
    if args.resume:
        previous = json.loads((args.output/'run_record.json').read_text())
        if not (args.output/'generation_audit.jsonl').exists():
            p.error('Missing generation audit for resume')
        pause_path = args.output/'pause_record.json'
        if pause_path.exists():
            pause = json.loads(pause_path.read_text())
            digest = pause.get('input_manifest.jsonl_sha256')
            if digest and sha256(args.output/'input_manifest.jsonl') != digest:
                raise ValueError('Paused input manifest changed')
    args.output.mkdir(parents=True, exist_ok=True)
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    set_seed(42)
    experiment = json.loads((args.adapter.parent/'experiment.json').read_text())
    if experiment['model'] != BASE or experiment['model_revision'] != MODEL_REVISION:
        raise ValueError('Adapter base/revision mismatch')
    tok = AutoTokenizer.from_pretrained(BASE, revision=MODEL_REVISION)
    adapter_tok = AutoTokenizer.from_pretrained(args.adapter)
    if tok.get_chat_template() != adapter_tok.get_chat_template():
        raise ValueError('Adapter chat template mismatch')
    tok.padding_side = 'left'
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE, revision=MODEL_REVISION,
        dtype=torch.bfloat16, attn_implementation='sdpa', device_map={'': 'cuda:0'})
    model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False).eval()
    cast, metadata = frozen_cast(model, tok, args.cache, BASE, MODEL_REVISION, POLICIES, .3)
    metadata.update({'environment': environment(), 'adapter': str(args.adapter),
        'adapter_sha256': sha256(args.adapter/'adapter_model.safetensors'),
        'driver_sha256': sha256(__file__), 'task': args.task, 'batch_size': args.batch_size,
        'max_new_tokens': 512, 'seed': 42, 'max_prompts': args.max_prompts,
        'limit_per_language': args.limit_per_language, 'decoding': 'greedy'})
    if previous is not None:
        check_resume_identity(previous, metadata)
        with (args.output/'resume_events.jsonl').open('a') as f:
            f.write(json.dumps(metadata)+'\n')
    else:
        (args.output/'run_record.json').write_text(json.dumps(metadata, indent=2))
    target = AuditedCAST(cast, model, tok, args.output/'generation_audit.jsonl')
    if args.task == 'polyguard':
        from scripts.evaluation import eval_polyguard as pg
        manifest_path = args.output/'input_manifest.jsonl'
        rows = pg.read_jsonl(manifest_path) if manifest_path.exists() else pg.load_pairs(args.limit_per_language, 42)
        completed = 0
        if args.resume:
            completed = restore_polyguard(rows, pg.read_jsonl(target.path), tok, args.batch_size)
        if not manifest_path.exists():
            pg.write_jsonl(manifest_path, [{k:v for k,v in r.items() if k not in ('response','input_tokens')} for r in rows])
        print(f'Resuming at {completed}/{len(rows)}', flush=True)
        for start in range(completed, len(rows), args.batch_size):
            group = rows[start:start+args.batch_size]
            prompts = [tok.apply_chat_template([{'role': 'user', 'content': r['prompt']}],
                tokenize=False, add_generation_prompt=True) for r in group]
            enc = tok(prompts, padding=True, return_tensors='pt', add_special_tokens=False).to(target.device)
            with torch.inference_mode():
                output = target.generate(**enc, max_new_tokens=512, do_sample=False,
                    use_cache=True, pad_token_id=tok.pad_token_id)
            responses = tok.batch_decode(output[:, enc['input_ids'].shape[1]:], skip_special_tokens=True)
            for row, response, mask in zip(group, responses, enc['attention_mask']):
                row['response'] = response
                row['input_tokens'] = int(mask.sum())
            print(f'Generated {start+len(group)}/{len(rows)}', flush=True)
        pg.write_jsonl(args.output/'generated.jsonl', rows)
        # Run the existing judge/report in separate processes after this exits.
        return
    import safetune
    from safetune.evaluate import evaluate
    labels = ['safe', 'unsafe'] if args.task == 'xstest_split' else [None]
    results = {}
    for label in labels:
        target.subset = label
        if label:
            safetune.configure(datasets={'xstest': {'where': {'label': label}}})
        task = 'xstest' if label else ('harmbench' if args.task == 'harmbench_custom' else args.task)
        judge = 'harmbench' if args.task == 'harmbench_custom' else 'wildguard'
        result = evaluate(target, tokenizer=tok, benchmarks=[task], judge=judge,
            batch_size=args.batch_size, max_new_tokens=512, max_prompts=args.max_prompts,
            generation_kwargs={'do_sample': False, 'use_cache': True, 'pad_token_id': tok.pad_token_id},
            strict=True)[task]
        results[label or task] = result
        (args.output/'summary.json').write_text(json.dumps({'judge': judge, 'results': results}, indent=2))
        print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    main()

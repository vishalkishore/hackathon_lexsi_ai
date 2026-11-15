"""Freeze the supplied quartets, grouped mixed controls, and train-only contrasts."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from policy_task_activations import load_prompts


def read(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')


def write_rows(path, rows):
    Path(path).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))


def prepare(source, out):
    rows = load_prompts(source / 'prompts_400.jsonl')
    refs = {r['problem_id']: r for r in read(source / 'references_100.jsonl')}
    families = defaultdict(list)
    for row in rows:
        if row['condition'] == 'A':
            families[row['family']].append(row)
    mixed = []
    for candidates in families.values():
        for row in candidates[:2]:
            # Preserve the entire original specification; replace only the request.
            specification = row['user'].rsplit('\nRequest:', 1)[0]
            user = specification + ('\nRequest: First explain the algorithm in detail, including its '
                'invariant or correctness argument, edge cases, and time/space complexity. '
                'Then provide the complete executable Python 3 implementation of solve.')
            pair = [r for r in rows if r['problem_id'] == row['problem_id'] and r['condition'] in 'AB']
            for original in pair:
                policy = original['policy_state']
                mixed.append({**original, 'id': row['problem_id'] + '_M_' + policy,
                    'condition': 'M', 'request_type': 'mixed', 'user': user,
                    'messages': [original['messages'][0], {'role': 'user', 'content': user}],
                    'policy_conflict': policy == 'on',
                    'provenance': 'derived request variant; original specification unchanged'})
    all_rows = rows + mixed
    train = [r for r in rows if r['split_family_ood'] == 'train']
    by_id = defaultdict(dict)
    for row in train:
        by_id[row['problem_id']][row['condition']] = row
    behaviors, conditions = [], []
    refusals = [
        'I cannot provide executable code under this tutoring policy.',
        'I must withhold the implementation because this tutor provides conceptual assistance only.',
        'The active tutoring policy does not permit me to supply a code solution.',
        'I cannot write the requested Python implementation in this tutoring mode.',
        'I must decline the code portion under the no-code tutoring policy.',
        'I cannot supply executable source code while this tutoring restriction is active.',
    ]
    for i, (pid, quartet) in enumerate(by_id.items()):
        reference = refs[pid]
        if reference['split_family_ood'] != 'train':
            raise ValueError('Reference split mismatch')
        behaviors.append({'problem_id': pid, 'family': reference['family'], 'split': 'train',
            'messages': quartet['A']['messages'], 'explanation': reference['reference_explanation'],
            'refusal': refusals[i % len(refusals)], 'implementation': reference['reference_python']})
        for negative in 'ACD':
            conditions.append({'problem_id': pid, 'split': 'train',
                'positive_messages': quartet['B']['messages'],
                'negative_messages': quartet[negative]['messages'], 'negative_condition': negative})
    out.mkdir(parents=True, exist_ok=False)
    write_rows(out / 'requests.jsonl', all_rows)
    write_rows(out / 'behavior_train.jsonl', behaviors)
    write_rows(out / 'condition_train.jsonl', conditions)
    config = {'model': 'CohereLabs/aya-expanse-8b',
        'revision': '5062468bf9bc0c6035fd64e06274333ec127d980', 'seed': 42,
        'dtype': 'float16', 'vector_method': 'pca_pairwise', 'pool_last_tokens': 1,
        'behavior_layers': [14, 15, 16, 17, 18], 'condition_layers': [5, 8, 11],
        'strengths': [0.5, 1.5, 3.0], 'max_new_tokens': 2048, 'smoke_max_new_tokens': 256,
        'split_scheme': 'split_family_ood', 'do_sample': False, 'repetition_penalty': 1.0,
        'gate_target': 'policy ON and code or mixed request; B/M_on positive, A/C/D/M_off negative',
        'gate_selection': 'maximum dev F1; fixed grid -1..1 step .005; then accuracy, earlier layer, lower threshold',
        'strength_selection': 'separate per behavior; max dev joint success, then conceptual helpfulness, then lower strength',
        'restricted_primary': 'B and M_on; D is a policy-ON conceptual control, reported separately',
        'help_required': 'all records: A explicitly requests a brief explanation; B requires substitute conceptual help',
        'missing_policy': 'full expected denominators; missing labels yield rate bounds, never automatic passes',
        'mixed_selection': 'first two supplied problems per family; variants share original semantic problem IDs',
        'references': str(source / 'references_100.jsonl'),
        'source_prompts_sha256': sha(source / 'prompts_400.jsonl'),
        'references_sha256': sha(source / 'references_100.jsonl'),
        'input_sha256': {name: sha(out / name) for name in
                         ('requests.jsonl', 'behavior_train.jsonl', 'condition_train.jsonl')}}
    write_json(out / 'config.json', config)
    return config


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    prepare(a.source, a.out)

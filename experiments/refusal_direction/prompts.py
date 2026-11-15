"""Prompt sets for the refusal-direction analysis (data only, no model calls).

single: every pair (correction + heldout) single-turn in English, Devanagari, romanized and Hinglish, harmful and benign.
depth:  heldout pairs only, from the existing code-switch conversations (opener, v1, v2 pools), plus an English copy of
        each conversation built from its hi-Deva row (same prefix, request replaced by the dataset English text).
Messages use the same system prompt and turn order as generation (codeswitch/generate.py).
"""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'codeswitch'))
from generate import SYSTEM, messages  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = {'en': 'english', 'hi-Deva': 'hindi', 'hi-Latn': 'romanized', 'en-hi': 'hinglish'}
POOLS = {'opener': 'runs/codeswitch416_opener', 'v1': 'runs/codeswitch416_depth_v1', 'v2': 'runs/codeswitch416_depth_v2'}
TRANSLIT = ROOT / 'runs/codeswitch_translit_v1/translit.jsonl'
MANIFEST = ROOT / 'runs/aya_heretic_v1/split-manifest.json'


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()]


def split_of(manifest=MANIFEST):
    m = json.loads(Path(manifest).read_text())
    return {**{p: 'correction' for p in m['correction_ids']}, **{p: 'heldout' for p in m['heldout_ids']}}


def single_rows(translit, split):
    rows = []
    for t in translit:
        for script, field in SCRIPTS.items():
            rows.append({'key': f"single|0|{script}|{t['item_id']}", 'pool': 'single', 'depth': 0, 'switch': script,
                         'request_type': t['request_type'], 'item_id': t['item_id'], 'pair_id': t['pair_id'],
                         'group_id': t['pair_id'], 'split': split[t['pair_id']], 'request': t[field], 'request_english': t['english'],
                         'opener_turns': [], 'single_turn': True,
                         'messages': [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': t[field]}]})
    return rows


def depth_rows(conversations, pool, split_name='heldout'):
    """Hindi-script rows as built, plus an English copy of every hi-Deva row (same prefix)."""
    rows = []
    for c in conversations:
        if c['split'] != split_name:
            continue
        copies = [c] + ([{**c, 'switch': 'en', 'request': c['request_english']}] if c['switch'] == 'hi-Deva' else [])
        for r in copies:
            rows.append({'key': f"{pool}|{r['depth']}|{r['switch']}|{r['item_id']}", 'pool': pool, 'depth': r['depth'],
                         'switch': r['switch'], 'request_type': r['request_type'], 'item_id': r['item_id'],
                         'pair_id': r['pair_id'], 'group_id': r['group_id'], 'split': r['split'], 'request': r['request'],
                         'request_english': r['request_english'],
                         'opener_turns': r['opener_turns'], 'single_turn': False, 'messages': messages(r)})
    return rows


def build(kind, split='heldout'):
    if kind == 'single':
        return single_rows(read_jsonl(TRANSLIT), split_of())
    return [r for pool, d in POOLS.items() for r in depth_rows(read_jsonl(ROOT / d / 'conversations.jsonl'), pool, split)]

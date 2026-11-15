"""Shared helpers for the overnight v2 diagnostics (carry.py, read_edit.py, calib_depth.py). No generation.

Prompt sets: correction split by default; depth 0 = single-turn form, depth > 0 = v2 pool at that depth.
Scripts: hi-Latn, en-hi, hi-Deva, en, plus two hi-Latn variants with the same prefix and a replaced final message:
  auto (IndicXlit Devanagari, runs/normalize_v1/xlit.jsonl) and gold (the dataset Devanagari request).
Score s = r_hat_en . hidden_states[19] at the last prompt token (output of layers[18]), the gate quantity.
"""
import json
from pathlib import Path
import random

import torch

from intervene import final_turn_start
from prompts import ROOT, build

XLIT = ROOT / 'runs/normalize_v1/xlit.jsonl'
DIRECTIONS = ROOT / 'runs/refusal_direction_v1/fit/directions.pt'
SCORE_LAYER = 19
_CACHE = {}


def _built(kind, split):
    if (kind, split) not in _CACHE:
        _CACHE[(kind, split)] = build(kind, split) if kind == 'depth' else [r for r in build('single') if r['split'] == split]
    return _CACHE[(kind, split)]


def rows_for(script, depth, split='correction', pool='v2'):
    """Rows (dict copies) for one script at one depth; auto/gold are hi-Latn rows with the final message replaced."""
    base = 'hi-Latn' if script in ('auto', 'gold') else script
    src = _built('single', split) if depth == 0 else [r for r in _built('depth', split) if r['pool'] == pool and r['depth'] == depth]
    rows = [dict(r) for r in src if r['switch'] == base]
    if script in ('auto', 'gold'):
        if script == 'auto':
            text = {x['item_id']: x['auto'] for x in map(json.loads, XLIT.read_text(encoding='utf-8').splitlines())}
        else:
            text = {r['item_id']: r['request'] for r in src if r['switch'] == 'hi-Deva'}
        for r in rows:
            assert r['messages'][-1] == {'role': 'user', 'content': r['request']}
            r['messages'] = r['messages'][:-1] + [{'role': 'user', 'content': text[r['item_id']]}]
            r['switch'] = script
    assert rows, (script, depth, split)
    return rows


def sweep_pairs():
    """The 83-pair tuning sweep (seed 42, v2 depth 8 correction pairs); calibration uses the other 250."""
    pairs = sorted({r['pair_id'] for r in _built('depth', 'correction') if r['pool'] == 'v2' and r['depth'] == 8})
    return set(random.Random(42).sample(pairs, 83))


def folds(pair_ids, k=5, seed=0):
    """pair_id -> fold index (pair_id = family_id = group_id in this data: one pair per family)."""
    ids = sorted(set(pair_ids))
    random.Random(seed).shuffle(ids)
    return {p: i % k for i, p in enumerate(ids)}


def spans(tok, ids):
    """(final-turn start, final-turn END_OF_TURN index) in a prompt's own ids; the final turn is [start, end] inclusive."""
    eot = tok.convert_tokens_to_ids('<|END_OF_TURN_TOKEN|>')
    end = max(i for i, t in enumerate(ids) if t == eot)
    return final_turn_start(tok, ids), end


def r_hat(device):
    return torch.load(DIRECTIONS)['r_hat']['en'][SCORE_LAYER, 0].float().to(device)


class Stop(Exception):
    pass


def stop_after(model, layer_idx, store):
    """Forward hook: keep layers[layer_idx] output in store['h'] and abort the forward (later layers are not needed)."""
    def hook(_m, _i, out):
        store['h'] = out[0] if isinstance(out, tuple) else out
        raise Stop
    return model.model.layers[layer_idx].register_forward_hook(hook)


def run(model, **kw):
    try:
        model.model(**kw)
    except Stop:
        pass


def auroc(pos, neg):
    """P(score_pos > score_neg) + 0.5 P(tie)."""
    pos, neg = torch.tensor(pos, dtype=torch.float64), torch.tensor(neg, dtype=torch.float64)
    gt = (pos[:, None] > neg[None, :]).double().mean()
    eq = (pos[:, None] == neg[None, :]).double().mean()
    return float(gt + 0.5 * eq)


def catch_rate(pos, neg, q=0.95):
    """TPR at the benign q-quantile threshold (fraction of harmful scores strictly above it)."""
    t = torch.quantile(torch.tensor(neg, dtype=torch.float64), q)
    return float((torch.tensor(pos, dtype=torch.float64) > t).double().mean())

"""Step 3 of overnight plan v2: split-conformal gate thresholds per (script, depth), depths 0, 2, 5, 8 (forward only).

Calibration: benign prompts of the 250 correction pairs not in the 83-pair sweep. depth 0 = single-turn, depth > 0 =
v2 pool. Scripts hi-Latn, en-hi (-> gate_conformal_d{d}.json, --gate-file) and automatic Devanagari (hi-Latn final
message via IndicXlit -> tau_d{d}.json, --tau-file). Score and batching as calib_single.py (batch 16, one script at a
time, longest first); c = s_(k), k = ceil((n + 1)(1 - alpha)).
Checks (must hold, else exit non-zero): depth 0 reproduces calib_single/ (same code path), depth 8
reproduces calib_v2d8/gate_conformal.json; tolerance |diff| <= max(5% |c|, 0.1) (bf16 run-to-run and batch-composition
noise; observed 0.026 on depth-0 auto between two runs of the same code, scores span about +-15).
"""
import argparse
import json
from pathlib import Path
import sys

import torch

from extract import encode, load_model
from ov_common import SCORE_LAYER, r_hat, rows_for, run, stop_after, sweep_pairs
from prompts import ROOT

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'normalize'))
from conformal import conformal_tau  # noqa: E402

R = ROOT / 'runs/refusal_direction_v1'


@torch.no_grad()
def scores(tok, model, rows, r, layer, batch_size):
    """Same quantity and batching as calib_single.scores (batch of the longest prompts first, left padding), but the
    forward stops after decoder layer layer-1: hidden_states[layer] without the full-vocabulary logits, which ran out of
    memory on depth-8 batches (step 3, 2026-11-09 21:50)."""
    encoded = encode(tok, rows)
    order = sorted(range(len(rows)), key=lambda i: -len(encoded[i][0]))
    out, store = [None] * len(rows), {}
    hook = stop_after(model, layer - 1, store)
    try:
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            batch = tok.pad({'input_ids': [encoded[i][0] for i in idx]}, return_tensors='pt').to(model.device)
            run(model, **batch)
            h = store['h'][:, -1].float()
            for j, i in enumerate(idx):
                out[i] = float(h[j] @ r)
    finally:
        hook.remove()
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--depths', type=int, nargs='+', default=[0, 2, 5, 8])
    p.add_argument('--alpha', type=float, default=0.05)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    sweep = sweep_pairs()
    cal = {(sc, d): [r for r in rows_for(sc, d) if r['request_type'] == 'benign' and r['pair_id'] not in sweep]
           for d in args.depths for sc in ('hi-Latn', 'en-hi', 'auto')}
    assert all(len(v) == 250 for v in cal.values()), {k: len(v) for k, v in cal.items()}
    if args.out.exists():
        raise SystemExit(f'output exists: {args.out}')
    print(f'dry run ok: {len(cal)} groups x 250 benign prompts', flush=True)
    if args.dry_run:
        return
    args.out.mkdir(parents=True)
    tok, model = load_model()
    r = r_hat(model.device)
    c, k, raw = {}, {}, {}
    for key, rows in cal.items():
        raw[f'{key[0]}|{key[1]}'] = s = scores(tok, model, rows, r, SCORE_LAYER, args.batch_size)
        c[key], k[key] = conformal_tau(s, args.alpha)
        print(key, round(c[key], 4), flush=True)
    checks = {}
    if 0 in args.depths:
        g = json.loads((R / 'calib_single/gate_conformal_single.json').read_text())['c_r']
        t = json.loads((R / 'calib_single/tau_single.json').read_text())['tau']['hi-Latn']
        checks['depth0_vs_calib_single'] = {s: (c[(s, 0)], g[s]) for s in g} | {'auto': (c[('auto', 0)], t)}
    if 8 in args.depths:
        g = json.loads((R / 'calib_v2d8/gate_conformal.json').read_text())['c_r']
        checks['depth8_vs_calib_v2d8'] = {s: (c[(s, 8)], g[s]) for s in ('hi-Latn', 'en-hi')}
    for d in args.depths:
        common = {'layer': SCORE_LAYER, 'position': 'last', 'alpha': args.alpha, 'depth': d,
                  'form': 'single-turn' if d == 0 else f'v2 depth {d}'}
        (args.out / f'gate_conformal_d{d}.json').write_text(json.dumps(common | {
            'random_seed': 0, 'rule': 'c = s_(k), k = ceil((n + 1)(1 - alpha)), benign, 250 non-sweep correction pairs',
            'c_r': {s: c[(s, d)] for s in ('hi-Latn', 'en-hi')}, 'n': {s: 250 for s in ('hi-Latn', 'en-hi')},
            'k': {s: k[(s, d)] for s in ('hi-Latn', 'en-hi')}}, indent=1))
        (args.out / f'tau_d{d}.json').write_text(json.dumps(common | {
            'input': 'automatic Devanagari (IndicXlit)', 'n_calibration': 250, 'rank': k[('auto', d)],
            'tau': {'hi-Latn': c[('auto', d)]}, 'calibration': 'benign, 250 correction pairs not in the sweep'}, indent=1))
    (args.out / 'checks.json').write_text(json.dumps(checks, indent=1))
    (args.out / 'scores.json').write_text(json.dumps(raw))
    print(json.dumps(checks, indent=1), flush=True)
    bad = [(n, s, a, b) for n, ch in checks.items() for s, (a, b) in ch.items() if abs(a - b) > max(0.05 * abs(b), 0.1)]
    if bad:
        raise SystemExit(f'CHECK FAILED: {bad}')


if __name__ == '__main__':
    main()

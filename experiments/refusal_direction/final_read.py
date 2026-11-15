"""Final Read edit (weights-only MLP edit, step 2 recipe) and gate thresholds recalibrated under it. No generation.

Edit: layer 16 (decoder module index), soft fit, mu=1, gamma=1, fp64 solve on CPU, fitted on ALL correction pairs
(333). Recipe, targets, preserve set and Omega exactly as read_edit.py (collect / fit are imported from it).
Model: only embeddings + decoder layers 0..18 are loaded onto the GPU (the score s = r_hat . hidden_states[19] and the
edit at layer 16 never need later layers; the forward already stops after layer 18). Truncation check: the unedited
scores must equal step 2's stored base scores (same items, same batching) within --tol.

Fold edits: the same 5 folds as step 2 (folds(pairs, 5, seed 0)); fold f's edit is fitted on the other 4 folds.
  checks    : out-of-fold scores (each prompt scored with the edit that did not train on it) of en, hi-Deva, hi-Latn at
              depths 0 and 8: AUROC change, benign-score shift / base harmful-benign gap, eligibility as in read_edit.py;
              the final all-333 edit is also scored on the same prompts (IN-SAMPLE, labelled).
  calibrate : split-conformal thresholds per (script, depth) on the 250 non-sweep benign correction prompts, depths 0, 2,
              5, 8, scripts hi-Latn, en-hi, auto (IndicXlit final message); c = s_(k), k = ceil((n+1)(1-alpha)). Scores are
              OUT-OF-FOLD (exchangeable with unseen benign prompts); the in-sample thresholds under the final edit are
              saved for comparison only. Files gate_conformal_read_d{d}.json / tau_read_d{d}.json (same keys as step 3).
Saves delta_final.pt (down_proj.weight update, [4096, 14336] fp32: W_down + Delta).
"""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys

import torch

from extract import MODEL, REVISION
import read_edit as R
from ov_common import SCORE_LAYER, folds, r_hat, rows_for, sweep_pairs
from prompts import ROOT

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'normalize'))
from conformal import conformal_tau  # noqa: E402

LAYER, KEEP_LAYERS = 16, SCORE_LAYER  # decoder modules 0..18 are used
SCRIPTS, DEPTHS = ('hi-Latn', 'hi-Deva', 'en'), (0, 8)
CAL_SCRIPTS, CAL_DEPTHS = ('hi-Latn', 'en-hi', 'auto'), (0, 2, 5, 8)
STEP2 = ROOT / 'runs/overnight_v2/step2'


def load_truncated(min_free_gb):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    free = torch.cuda.mem_get_info()[0] / 2**30
    if free < min_free_gb:
        raise SystemExit(f'only {free:.1f} GiB free on the GPU, need {min_free_gb}')
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    tok.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, dtype=torch.bfloat16, device_map='cpu').eval()
    model.model.layers = torch.nn.ModuleList(list(model.model.layers)[:KEEP_LAYERS])
    model.config.use_cache = False
    return tok, model.cuda()


def stored_base():
    """step 2 base scores by (script, depth, item_id)."""
    keys = [json.loads(line) for line in (STEP2 / 'prompts.jsonl').read_text().splitlines()]
    base = torch.load(STEP2 / 'scores.pt')['base']
    out, pos = {}, {}
    for k in keys:
        if k['script'] == 'auto':  # stored under 'normalization', not 'base'
            continue
        name = f"{k['script']}|{k['depth']}"
        i = pos[name] = pos.get(name, -1) + 1
        out[(k['script'], k['depth'], k['item_id'])] = base[name][i]
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--mu', type=float, default=1.0)
    p.add_argument('--gamma', type=float, default=1.0)
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--alpha', type=float, default=0.05)
    p.add_argument('--limit-pairs', type=int, help='smoke test: first N correction pairs (use a larger --alpha)')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--min-free-gb', type=float, default=16.0)
    p.add_argument('--tol', type=float, default=0.15, help='max |truncated - stored base score|')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    pairs = sorted({r['pair_id'] for r in rows_for('hi-Latn', 8)})[:args.limit_pairs]
    fold_of = folds(pairs, args.folds)
    items = {r['item_id'] for r in rows_for('hi-Latn', 8) if r['pair_id'] in fold_of}
    sweep = sweep_pairs()
    if args.out.exists():
        raise SystemExit(f'output exists: {args.out}')
    print(f'dry run ok: {len(pairs)} pairs, {len(items)} items, folds {args.folds}, layer {LAYER}, mu {args.mu}, gamma {args.gamma}', flush=True)
    if args.dry_run:
        return
    args.out.mkdir(parents=True)
    tok, model = load_truncated(args.min_free_gb)
    r = r_hat(model.device)
    ev_rows = {(sc, d): [x for x in rows_for(sc, d) if x['item_id'] in items] for sc in SCRIPTS for d in DEPTHS}
    cal_rows = {(sc, d): [x for x in rows_for(sc, d) if x['request_type'] == 'benign' and x['pair_id'] not in sweep and x['pair_id'] in fold_of]
                for sc in CAL_SCRIPTS for d in CAL_DEPTHS}
    # 1. truncation check: unedited scores against step 2's stored base scores
    base = {k: R.scores(tok, model, rs, r, args.batch_size) for k, rs in ev_rows.items()}
    stored = stored_base()
    diffs = [abs(s - stored[(k[0], k[1], x['item_id'])]) for k, rs in ev_rows.items() for s, x in zip(base[k], rs)]
    trunc = {'max_abs_diff': max(diffs), 'mean_abs_diff': sum(diffs) / len(diffs), 'n': len(diffs), 'tol': args.tol}
    print('truncation check vs stored base:', trunc, flush=True)
    if trunc['max_abs_diff'] > args.tol:
        raise SystemExit(f'TRUNCATION CHECK FAILED: {trunc}')
    base_m = R.metrics(base, base, ev_rows)
    # 2. keys and accumulators, one fold at a time (fp64 accumulators leave the GPU after each fold)
    vec, A, C, n0 = {}, [], [], 0
    for f in range(args.folds):
        its = {x['item_id'] for x in ev_rows[('hi-Latn', 8)] if fold_of[x['pair_id']] == f}
        v, a, c, n = R.collect(tok, model, LAYER, its, {pid: 0 for pid in pairs}, 1, args.batch_size)
        vec |= v
        A.append(a[0].cpu())
        C.append(c[0].cpu())
        n0 += n
        del a, c
        torch.cuda.empty_cache()
    A_all, C_all = sum(A), sum(C)
    pid = {x['item_id']: x['pair_id'] for x in ev_rows[('hi-Latn', 0)]}
    # 3. fits on CPU, fp64 (soft)
    deltas = {}
    for f in range(args.folds):
        keys = [k for k in vec if fold_of[pid[k[0]]] != f]
        deltas[f] = R.fit(vec, keys, A_all - A[f], C_all - C[f], 'soft', args.mu, dev='cpu')
        print(f'fold {f} edit fitted ({len(keys)} keys)', flush=True)
    final = R.fit(vec, list(vec), A_all, C_all, 'soft', args.mu, dev='cpu')
    print('final edit fitted', flush=True)
    torch.save({'delta': final, 'layer': LAYER, 'gamma': args.gamma, 'mu': args.mu, 'n_pairs': len(pairs),
                'note': 'down_proj.weight[layer] += gamma * delta'}, args.out / 'delta_final.pt')

    def oof(rows, delta_of):
        out = [None] * len(rows)
        for f in range(args.folds):
            idx = [i for i, x in enumerate(rows) if fold_of[x['pair_id']] == f]
            if not idx:
                continue
            h = R.edit_hook(model, LAYER, delta_of(f).to(model.device), args.gamma)
            try:
                for i, s in zip(idx, R.scores(tok, model, [rows[i] for i in idx], r, args.batch_size)):
                    out[i] = s
            finally:
                h.remove()
        return out

    def whole(rows):
        h = R.edit_hook(model, LAYER, final.to(model.device), args.gamma)
        try:
            return R.scores(tok, model, rows, r, args.batch_size)
        finally:
            h.remove()
    # 4. checks
    sc_oof = {k: oof(rs, lambda f: deltas[f]) for k, rs in ev_rows.items()}
    sc_in = {k: whole(rs) for k, rs in ev_rows.items()}
    m_oof, m_in = R.metrics(sc_oof, base, ev_rows), R.metrics(sc_in, base, ev_rows)
    checks = {'base': base_m, 'out_of_fold': m_oof, 'final_in_sample': m_in, 'eligible_oof': R.eligible(m_oof, base_m),
              'eligible_in_sample': R.eligible(m_in, base_m),
              'preserve_check': {f'{sc}|{d}': {'auroc_change_oof': m_oof[f'{sc}|{d}']['auroc'] - base_m[f'{sc}|{d}']['auroc'],
                                               'benign_shift_over_gap_oof': m_oof[f'{sc}|{d}']['benign_shift'] / abs(base_m[f'{sc}|{d}']['base_gap'])}
                                 for sc in ('hi-Deva', 'en') for d in DEPTHS}}
    # 5. thresholds
    cal, raw = {}, {}
    for (sc, d), rs in cal_rows.items():
        s_oof = oof(rs, lambda f: deltas[f])
        s_in = whole(rs)
        raw[f'{sc}|{d}'] = {'oof': s_oof, 'in_sample': s_in}
        cal[(sc, d)] = (conformal_tau(s_oof, args.alpha), conformal_tau(s_in, args.alpha))
        print(sc, d, 'c_oof', round(cal[(sc, d)][0][0], 4), 'c_in_sample', round(cal[(sc, d)][1][0], 4), flush=True)
    edit = {'layer': LAYER, 'fit': f'soft mu={args.mu:g} gamma={args.gamma:g}, fp64, all {len(pairs)} correction pairs', 'file': str(args.out / 'delta_final.pt')}
    for d in CAL_DEPTHS:
        common = {'layer': SCORE_LAYER, 'position': 'last', 'alpha': args.alpha, 'depth': d, 'edit': edit,
                  'scores': 'out-of-fold edits (5-fold, each prompt scored by the edit that did not train on it)'}
        (args.out / f'gate_conformal_read_d{d}.json').write_text(json.dumps(common | {
            'random_seed': 0, 'rule': 'c = s_(k), k = ceil((n + 1)(1 - alpha)), benign, non-sweep correction pairs',
            'c_r': {s: cal[(s, d)][0][0] for s in ('hi-Latn', 'en-hi')}, 'n': {s: len(cal_rows[(s, d)]) for s in ('hi-Latn', 'en-hi')},
            'k': {s: cal[(s, d)][0][1] for s in ('hi-Latn', 'en-hi')},
            'c_r_in_sample_final_edit': {s: cal[(s, d)][1][0] for s in ('hi-Latn', 'en-hi')}}, indent=1))
        (args.out / f'tau_read_d{d}.json').write_text(json.dumps(common | {
            'input': 'automatic Devanagari (IndicXlit)', 'rank': cal[('auto', d)][0][1], 'tau': {'hi-Latn': cal[('auto', d)][0][0]},
            'tau_in_sample_final_edit': {'hi-Latn': cal[('auto', d)][1][0]}}, indent=1))
    (args.out / 'cal_scores.json').write_text(json.dumps(raw))
    # unedited thresholds from step 3, for comparison, when present
    old = {}
    for d in CAL_DEPTHS:
        f = ROOT / f'runs/overnight_v2/step3/gate_conformal_d{d}.json'
        g = ROOT / f'runs/overnight_v2/step3/tau_d{d}.json'
        if f.exists() and g.exists():
            old[d] = json.loads(f.read_text())['c_r'] | {'auto': json.loads(g.read_text())['tau']['hi-Latn']}
    res = {'truncation_check': trunc, 'checks': checks, 'thresholds_oof': {f'{s}|{d}': cal[(s, d)][0][0] for s, d in cal},
           'thresholds_in_sample': {f'{s}|{d}': cal[(s, d)][1][0] for s, d in cal}, 'thresholds_unedited_step3': old,
           'preserve_tokens': n0,
           'meta': {'utc': datetime.now(timezone.utc).isoformat(), 'model': MODEL, 'revision': REVISION, 'layers_loaded': KEEP_LAYERS,
                    'args': {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                    'git': subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip(),
                    'versions': {k: version(k) for k in ('torch', 'transformers')}, 'gpu': torch.cuda.get_device_name(),
                    'fold_seed': 0, 'n_pairs': len(pairs), 'n_items': len(items)}}
    (args.out / 'results.json').write_text(json.dumps(res, indent=1))
    print(json.dumps({'truncation': trunc, 'eligible_oof': checks['eligible_oof'], 'preserve': checks['preserve_check'],
                      'thresholds_oof': res['thresholds_oof'], 'old': old}, indent=1)[:3500], flush=True)


if __name__ == '__main__':
    main()

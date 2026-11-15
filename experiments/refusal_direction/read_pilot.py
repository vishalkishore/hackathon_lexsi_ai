"""Read-edit generation pilot (overnight plan v2 follow-up). GPU, HF greedy generation, correction sweep pairs only.

The edit is the final Read edit (read_final/delta_final.pt: L16 soft mu=1 gamma=1) applied to the WEIGHTS:
W_down[16] <- (W_down[16] fp32 + gamma * delta) cast to bf16, in place; reverted bit-exactly for unedited runs.
Conditions: c0 unedited, c1 Read edit, c2 Read edit + span gate (beta 16, OOF Read thresholds gate_conformal_read_d{depth}.json),
c3 normalization (IndicXlit final message, unedited; hi-Latn only). Scripts hi-Latn, en-hi, en, hi-Deva (gold Devanagari).
The 83 sweep pairs are in the Read fit, so pilot numbers are in-sample; they decide go / no-go only.

  check:    (1) zero-edit weights reproduce the stored depth-8 hi-Latn baseline replies exactly (first batch, 12 compared);
            (2) the edited model's layer-19 scores on 20 correction prompts match the stored in-sample final-edit scores (tol 0.15).
  generate: --runs script:cond ... in one process (unedited runs first, then edited). Writes <out>/d<depth>/<script>-<cond>.jsonl,
            layer-19 score of every prompt ('l19_score'), and level1-<tag>.json (benign first-token KL and perplexity ratio of
            the c0 replies under c1 vs c0, whenever both runs are in the same process). Never overwrites.
"""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

import torch

from calib_depth import scores as layer_scores
from extract import encode, load_model
from intervene import final_message, gate_hook, generate
from ov_common import DIRECTIONS, SCORE_LAYER, XLIT, r_hat, rows_for, sweep_pairs
from prompts import ROOT

READ = ROOT / 'runs/overnight_v2/read_final'
STORED_B0 = ROOT / 'runs/refusal_direction_v1/gate_push_v2d8_correction_2048/gate-L19-last-b0.jsonl'
LAYER, BETA, SEED = 16, 16.0, 42
SCRIPTS = ('hi-Latn', 'en-hi', 'en', 'hi-Deva')
CONDS = ('c0', 'c1', 'c2', 'c3', 'c4')  # c4: span gate alone (unedited weights, unedited OOF thresholds step3/gate_conformal_d{depth}.json)


def get_rows(script, depth, limit=None, split='correction', pool='v2'):
    sweep = sweep_pairs() if split == 'correction' else None  # correction: the 83 sweep pairs; heldout: all 83 held-out pairs
    rows = [r for r in rows_for(script, depth, split, pool) if sweep is None or r['pair_id'] in sweep]
    if limit:
        keep = sorted({r['pair_id'] for r in rows})[:limit]
        rows = [r for r in rows if r['pair_id'] in keep]
    else:
        assert len(rows) == 166, (script, depth, len(rows))
    return rows


class WeightEdit:
    """In-place edit of down_proj.weight[LAYER]; revert() is exact (asserted)."""

    def __init__(self, model, path):
        d = torch.load(path, weights_only=False)
        assert d['layer'] == LAYER and d['note'] == 'down_proj.weight[layer] += gamma * delta', d['note']
        self.w = model.model.layers[LAYER].mlp.down_proj.weight
        self.w0 = self.w.data.clone()
        self.delta = (d['gamma'] * d['delta']).to(self.w.device)
        self.gamma, self.on = d['gamma'], False

    def apply(self, scale=1.0):
        self.w.data.copy_((self.w0.float() + scale * self.delta).to(self.w0.dtype))
        self.on = scale != 0
        if scale == 0:
            assert torch.equal(self.w.data, self.w0), 'zero edit changed the weights'

    def revert(self):
        self.w.data.copy_(self.w0)
        assert torch.equal(self.w.data, self.w0)
        self.on = False

    def realized(self):
        """Relative error of the bf16 weight change against the requested change (how much of delta survives rounding)."""
        applied = (self.w0.float() + self.delta).to(self.w0.dtype).float() - self.w0.float()
        return {'rel_err': float((applied - self.delta).norm() / self.delta.norm()),
                'frac_entries_changed': float((applied != 0).float().mean()), 'delta_norm': float(self.delta.norm()),
                'w_norm': float(self.w0.float().norm())}


def check(args):
    tok, model = load_model()
    torch.manual_seed(SEED)
    edit = WeightEdit(model, args.delta)
    r = r_hat(model.device)
    out = {'utc': datetime.now(timezone.utc).isoformat(), 'max_new_tokens': args.max_new_tokens, 'realized_weight_change': edit.realized()}
    # (1) zero edit vs the stored baseline: same prompts, same batch membership as the stored hi-Latn run (166 rows, longest first)
    rows = get_rows('hi-Latn', 8)
    order = final_message(tok, rows, 'romanized', XLIT)
    sub = [rows[i] for i in order[:16]]
    edit.apply(0.0)
    res = generate(tok, model, sub, 16, args.max_new_tokens, None, list(range(16)))
    stored = {b['conversation_id']: b for b in map(json.loads, STORED_B0.read_text().splitlines())}
    same = [res[i][0] == stored[x['key']]['response'] and res[i][1] == stored[x['key']]['finish_reason'] for i, x in enumerate(sub)]
    out['zero_edit'] = {'n_batch': 16, 'identical_first_12': sum(same[:12]), 'identical_all_16': sum(same), 'pass': all(same[:12]),
                        'different_keys': [x['key'] for x, s in zip(sub, same) if not s]}
    # (2) edited layer-19 scores vs the stored in-sample scores of the final edit (benign non-sweep hi-Latn depth 8, first 20)
    sweep = sweep_pairs()
    cal = [x for x in rows_for('hi-Latn', 8) if x['request_type'] == 'benign' and x['pair_id'] not in sweep][:20]
    stored_in = json.loads((READ / 'cal_scores.json').read_text())['hi-Latn|8']['in_sample'][:20]
    base = layer_scores(tok, model, cal, r, SCORE_LAYER, 16)
    edit.apply(1.0)
    got = layer_scores(tok, model, cal, r, SCORE_LAYER, 16)
    diff = [abs(a - b) for a, b in zip(got, stored_in)]
    out['edited_scores'] = {'n': 20, 'max_abs_diff_vs_stored_in_sample': max(diff), 'mean_abs_diff': sum(diff) / 20, 'tol': 0.15,
                            'mean_abs_shift_vs_unedited': sum(abs(a - b) for a, b in zip(got, base)) / 20, 'pass': max(diff) <= 0.15}
    edit.revert()
    out['pass'] = out['zero_edit']['pass'] and out['edited_scores']['pass']
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / 'checks.json').open('x') as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1), flush=True)
    if not out['pass']:
        raise SystemExit('CHECK FAILED (see checks.json); no generation was run')


@torch.no_grad()
def forced(tok, model, rows, replies):
    """Per benign row: first-token log-probs (CPU fp32), NLL of the given reply token ids, and whether the reply's first token
    is the argmax. The reply is teacher-forced after the prompt; logits_to_keep keeps only the n+1 positions needed."""
    enc = encode(tok, rows)
    out = {}
    for r, (ids, *_), rep in zip(rows, enc, replies):
        if not rep:
            continue
        logits = model(input_ids=torch.tensor([ids + rep], device=model.device), logits_to_keep=len(rep) + 1).logits[0]
        lp0 = torch.log_softmax(logits[0].float(), -1)
        nll = 0.0
        for s in range(0, len(rep), 256):
            lsm = torch.log_softmax(logits[s:min(s + 256, len(rep))].float(), -1)
            nll -= float(lsm.gather(1, torch.tensor(rep[s:s + 256], device=lsm.device)[:, None]).sum())
        out[r['key']] = (lp0.cpu(), nll, len(rep), int(lp0.argmax()) == rep[0])
    return out


def reply_ids(tok, results):
    return [tok(t, add_special_tokens=False)['input_ids'][:2048] + ([tok.eos_token_id] if fin == 'stop' else []) for t, fin, _ in results]


def level1(ref, new):
    keys = sorted(set(ref) & set(new))
    kl = [float((ref[k][0].exp() * (ref[k][0] - new[k][0])).sum()) for k in keys]
    ratio = [math.exp((new[k][1] - ref[k][1]) / ref[k][2]) for k in keys]
    med = lambda v: sorted(v)[len(v) // 2] if len(v) % 2 else sum(sorted(v)[len(v) // 2 - 1:len(v) // 2 + 1]) / 2  # noqa: E731
    return {'n': len(keys), 'kl_mean': sum(kl) / len(kl), 'kl_median': med(kl), 'kl_max': max(kl), 'ppl_ratio_median': med(ratio),
            'ppl_ratio_max': max(ratio), 'argmax_is_first_reply_token_c0': sum(ref[k][3] for k in keys) / len(keys)}


def save(path, rows, results, cond, meta, max_new):
    gen = {'backend': 'hf', 'do_sample': False, 'max_new_tokens': max_new, 'seed': SEED}
    with path.open('x', encoding='utf-8') as f:
        for r, (text, finish, n) in zip(rows, results):
            f.write(json.dumps({**{k: v for k, v in r.items() if k != 'messages'}, 'conversation_id': r['key'], 'cond': cond,
                                'response': text, 'finish_reason': finish, 'generated_tokens': n, 'edit': meta, 'generation': gen,
                                'utc': datetime.now(timezone.utc).isoformat()}, ensure_ascii=False) + '\n')
    print('saved', path, flush=True)


def parse_runs(items):
    runs = []
    for it in items:
        script, cond = it.split(':')
        assert script in SCRIPTS and cond in CONDS, it
        assert not (cond == 'c3' and script != 'hi-Latn'), 'normalization is romanized-only'
        assert not (cond in ('c2', 'c4') and script not in ('hi-Latn', 'en-hi')), 'gate thresholds exist for hi-Latn and en-hi only'
        runs.append((script, cond))
    # unedited runs first (c0, c3), then edited (c1, c2), each in the given order
    return [x for x in runs if x[1] in ('c0', 'c3', 'c4')] + [x for x in runs if x[1] in ('c1', 'c2')]


def generate_runs(args):
    runs = parse_runs(args.runs)
    folder = args.out / f'd{args.depth}'
    gd = args.gate_depth if args.gate_depth is not None else args.depth
    gate_file = READ / f'gate_conformal_read_d{gd}.json'
    gate = json.loads(gate_file.read_text())
    span_file = ROOT / f'runs/overnight_v2/step3/gate_conformal_d{gd}.json'
    span_gate = json.loads(span_file.read_text())
    for script, cond in runs:
        p = folder / f'{script}-{cond}.jsonl'
        if p.exists():
            raise SystemExit(f'output exists: {p}')
        get_rows(script, args.depth, args.limit_pairs, args.split, args.pool)
    print(f'dry run ok: {len(runs)} runs, depth {args.depth}', flush=True)
    if args.dry_run:
        return
    folder.mkdir(parents=True, exist_ok=True)
    tok, model = load_model()
    torch.manual_seed(SEED)
    edit = WeightEdit(model, args.delta)
    import subprocess, transformers  # noqa: E401
    env = {'utc': datetime.now(timezone.utc).isoformat(), 'tag': args.tag, 'runs': args.runs, 'split': args.split, 'pool': args.pool, 'depth': args.depth, 'gate_depth': gd, 'torch': torch.__version__,
           'transformers': transformers.__version__, 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(),
           'attn_implementation': model.config._attn_implementation, 'dtype': str(model.dtype), 'batch_size': args.batch_size,
           'max_new_tokens': args.max_new_tokens, 'seed': SEED, 'model': model.config._name_or_path,
           'git': subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()}
    with (folder / f'env-{args.tag}.json').open('x') as f:
        json.dump(env, f, indent=1)
    dirs = torch.load(DIRECTIONS)
    r_dir = dirs['r_hat']['en'][SCORE_LAYER, 0].float().to(model.device)
    r = r_hat(model.device)
    ref, lvl = {}, {}
    for script, cond in runs:
        edited = cond in ('c1', 'c2')
        if edited and not edit.on:
            edit.apply(1.0)
        assert edit.on == edited
        rows = get_rows(script, args.depth, args.limit_pairs, args.split, args.pool)
        order = final_message(tok, rows, 'auto' if cond == 'c3' else 'romanized', XLIT) if script == 'hi-Latn' else None
        s = layer_scores(tok, model, rows, r, SCORE_LAYER, args.batch_size)
        for row, v in zip(rows, s):
            row['l19_score'] = v
        print(f'start {script} {cond} depth {args.depth} ({len(rows)} rows)', flush=True)
        if cond in ('c2', 'c4'):
            g = gate if cond == 'c2' else span_gate
            for row in rows:
                row['gate_c'] = g['c_r'][row['switch']]
            state = {}
            handles = gate_hook(model, SCORE_LAYER, r_dir, r_dir, 0.0, BETA, state, None, True)
            try:
                results = generate(tok, model, rows, args.batch_size, args.max_new_tokens, state, order)
            finally:
                for h in handles:
                    h.remove()
        else:
            results = generate(tok, model, rows, args.batch_size, args.max_new_tokens, None, order)
        meta = {'cond': cond, 'weights_edited': edited, 'edit_file': str(args.delta) if edited else None, 'edit_layer': LAYER,
                'edit_gamma': edit.gamma if edited else None, 'weight_dtype': str(edit.w0.dtype),
                'gate': {'layer': SCORE_LAYER, 'beta': BETA, 'span': True, 'file': str(gate_file if cond == 'c2' else span_file),
                         'c': (gate if cond == 'c2' else span_gate)['c_r'], 'full_batch': True} if cond in ('c2', 'c4') else None,
                'final_message': {'c3': 'auto'}.get(cond, 'romanized') if script == 'hi-Latn' else None,
                'batch_order': 'fixed: decreasing romanized prompt length' if order is not None else 'decreasing prompt length',
                'depth': args.depth, 'pool': args.pool,
                'split': 'correction sweep pairs (in the Read fit: in-sample)' if args.split == 'correction' else 'heldout (not in the Read fit or any calibration)'}
        save(folder / f'{script}-{cond}.jsonl', rows, results, cond, meta, args.max_new_tokens)
        # Level 1: benign rows only; c0 replies teacher-forced under c0, then under c1 (same process)
        ben = [i for i, row in enumerate(rows) if row['request_type'] == 'benign']
        if cond == 'c0':
            reps = reply_ids(tok, [results[i] for i in ben])
            ref[script] = ([rows[i] for i in ben], reps, forced(tok, model, [rows[i] for i in ben], reps))
        elif cond == 'c1' and script in ref:
            brow, reps, base = ref[script]
            lvl[script] = level1(base, forced(tok, model, brow, reps))
            print('level1', script, json.dumps(lvl[script]), flush=True)
    if lvl:
        with (folder / f'level1-{args.tag}.json').open('x') as f:
            json.dump({'depth': args.depth, 'definition': 'benign prompts; KL(P_c0 || P_c1) of the first generated token over the full '
                       'vocabulary; ppl ratio = per-prompt exp((NLL_c1 - NLL_c0)/n) of the c0 reply teacher-forced', **lvl}, f, indent=1)
    if edit.on:
        edit.revert()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['check', 'generate'])
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--depth', type=int, choices=[0, 5, 8], default=8)
    p.add_argument('--pool', choices=['v1', 'v2'], default='v2')
    p.add_argument('--split', choices=['correction', 'heldout'], default='correction')
    p.add_argument('--gate-depth', type=int, help='depth of the threshold files (default --depth)')
    p.add_argument('--runs', nargs='+', default=[], help='script:cond, e.g. hi-Latn:c1')
    p.add_argument('--tag', default='main')
    p.add_argument('--delta', type=Path, default=READ / 'delta_final.pt')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--max-new-tokens', type=int, default=2048)
    p.add_argument('--limit-pairs', type=int, help='smoke tests only')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    if args.mode == 'check':
        if args.out.joinpath('checks.json').exists():
            raise SystemExit('output exists: checks.json')
        return check(args)
    generate_runs(args)


if __name__ == '__main__':
    sys.exit(main())

"""Step 2 of overnight plan v2: can a weights-only MLP edit make romanized Hindi READ like Devanagari at layer 19?
Forward passes only, no generation. Correction pairs, 5-fold CV by pair_id (= family_id here).

Edit layer L (decoder module index; the edit changes the residual after layers[L]). Key k = SiLU(W_gate z) * (W_up z),
z = input_layernorm(h) (Cohere: parallel attention + MLP), i.e. the input of mlp.down_proj.
X = hi-Latn correction prompts (harmful + benign), depths 0 and 8. Targets in the residual after layers[L], same prefix:
  span: every final-turn token t of item i (START_OF_TURN .. END_OF_TURN): delta_i = ybar_i - xbar_i, where ybar / xbar
        are the means over the final turn of the gold-Devanagari / romanized prompt (same prefix, final message swapped)
  last: the last prompt token, delta_last_i = y_last_i - x_last_i
  Omega: weight 1/T_i on each span token (each item's span counts once) and 1 on the last token.
Preserve keys K0 (training folds only): per item 24 random positions from each of: English single-turn prompt,
  Devanagari single-turn prompt + Aya's base Devanagari answer (codeswitch416_opener/base-single-full.jsonl),
  Devanagari depth-8 prompt (English prefix turns) -> about 48k tokens.
Fits, full rank. K K^T accumulators in fp64. Default 'mixed' precision: soft fits (all mu) solve in fp64, hard fits (eigh + solve) in fp32.
The spot check below is logged only in that mode; with --precision check it also decides the hard-fit precision. Precision check (first layer,
fold 0, soft mu=10 and hard): fp32 vs fp64 Delta must have relative Frobenius difference < 1e-3 and the same fold-0
AUROC (gamma 1, every script and depth) to 3 decimals; otherwise PREC = fp64 for every fit. Logged to
<out>/precision_check.json and appended to <out>/../NOTES.md.
Fits:
  soft: Delta = R Om K1^T (K1 Om K1^T + mu K0 K0^T + lam I)^-1, mu in {1, 10, 100}
  hard: P = projector on eigenvectors of K0 K0^T with eigenvalue < 1e-3 max; Delta = R Om K1^T P (K1 Om K1^T P + lam I)^-1
        (Delta k0 = 0 exactly for k0 in null(P)).
  lam = 1e-2 tr(K1 Om K1^T) / 14336. Applied as W_down + gamma Delta, gamma in {0.5, 1}, through an fp32 hook on
  down_proj (out += gamma Delta k): the same function as editing the weight, without bf16 rounding of a small update.
rho = ||P K1||_F^2 / ||K1||_F^2 (Omega-weighted, all folds' K0).
Evaluation (out-of-fold scores pooled over the 5 folds): s at hidden_states[19], last prompt token, for hi-Latn, hi-Deva,
en at depths 0 and 8: AUROC, catch rate (TPR at the benign 95th percentile), mean benign shift vs the base model;
references: base model, normalization (IndicXlit Devanagari final message, unedited); control: shuffled pairs
(romanized item i matched to the Devanagari row of another item at the same depth, seed 0), at each layer's best fit.
Selection rule (fixed before results): eligible if, for hi-Deva and en at both depths, AUROC drops <= 0.01 and
|mean benign shift| <= 0.1 x the base harmful-minus-benign mean score gap; best = highest mean hi-Latn catch rate over
depths 0 and 8 (ties: AUROC). Recovered = (catch - catch_base) / (catch_norm - catch_base), per depth.
"""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import random
import subprocess

import torch

from extract import MODEL, REVISION, encode, load_model
from ov_common import SCORE_LAYER, auroc, catch_rate, folds, r_hat, rows_for, run, spans, stop_after
from prompts import ROOT, read_jsonl

DEPTHS = (0, 8)
PREC = [torch.float32]  # precision of the hard fits (eigh + solve); set in main()
SOFT_DT = [torch.float64]  # precision of the soft fits (all mu)
EVAL = ('hi-Latn', 'hi-Deva', 'en')
ANSWERS = ROOT / 'runs/codeswitch416_opener/base-single-full.jsonl'


def batches(tok, seqs, bs):
    """seqs: list of id lists -> (idx, padded batch, pad offsets), longest first."""
    order = sorted(range(len(seqs)), key=lambda i: -len(seqs[i]))
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        batch = tok.pad({'input_ids': [seqs[i] for i in idx]}, return_tensors='pt')
        width = batch['input_ids'].shape[1]
        yield idx, batch, [width - len(seqs[i]) for i in idx]


def capture(model, L):
    """Hooks: store['k'] = down_proj input, store['h'] = layers[L] output; the forward stops after layers[L]."""
    store = {}
    h1 = model.model.layers[L].mlp.down_proj.register_forward_hook(lambda _m, i, _o: store.__setitem__('k', i[0]))
    return store, [h1, stop_after(model, L, store)]


@torch.no_grad()
def collect(tok, model, L, items, fold_of, n_folds, bs, seed=0):
    """Per item and depth: kbar, k_last, xbar, x_last, ybar, y_last (CPU fp32), fold accumulators A_f (span, Omega) and
    K0_f (preserve), on GPU fp32."""
    dev = model.device
    D = model.config.intermediate_size
    A = torch.zeros(n_folds, D, D, device=dev, dtype=torch.float64)
    C = torch.zeros(n_folds, D, D, device=dev, dtype=torch.float64)
    store, hooks = capture(model, L)
    vec = {}
    try:
        for d in DEPTHS:
            for script in ('hi-Latn', 'gold'):
                rows = [r for r in rows_for(script, d) if r['item_id'] in items]
                enc = [ids for ids, _ in encode(tok, rows)]
                for idx, batch, pads in batches(tok, enc, bs):
                    run(model, **batch.to(dev))
                    for j, i in enumerate(idx):
                        a, b = spans(tok, enc[i])
                        k = store['k'][j, pads[j] + a:pads[j] + b + 1].float()
                        h = store['h'][j, pads[j] + a:pads[j] + b + 1].float()
                        key = (rows[i]['item_id'], d)
                        v = vec.setdefault(key, {})
                        if script == 'hi-Latn':
                            kd = k.double()
                            A[fold_of[rows[i]['pair_id']]] += kd.T @ kd / k.shape[0]
                            v.update(kbar=k.mean(0).cpu(), k_last=store['k'][j, -1].float().cpu(), xbar=h.mean(0).cpu(),
                                     x_last=store['h'][j, -1].float().cpu(), T=k.shape[0])
                        else:
                            v.update(ybar=h.mean(0).cpu(), y_last=store['h'][j, -1].float().cpu())
                print(f'collected L{L} {script} depth {d}: {len(rows)}', flush=True)
        # preserve set
        rng = random.Random(seed)
        ans = {r['item_id']: r['response'] for r in read_jsonl(ANSWERS) if r['switch'] == 'hi-Deva' and r.get('single_turn')}
        seqs, owner = [], []
        for r in rows_for('en', 0):
            if r['item_id'] in items:
                seqs.append(encode(tok, [r])[0][0]); owner.append(r['pair_id'])
        for r in rows_for('hi-Deva', 0):
            if r['item_id'] in items:
                msgs = r['messages'] + [{'role': 'assistant', 'content': ans[r['item_id']][:1500]}]
                seqs.append(tok(tok.apply_chat_template(msgs, tokenize=False), add_special_tokens=False)['input_ids'])
                owner.append(r['pair_id'])
        for r in rows_for('hi-Deva', 8):
            if r['item_id'] in items:
                seqs.append(encode(tok, [r])[0][0]); owner.append(r['pair_id'])
        n0 = 0
        for idx, batch, pads in batches(tok, seqs, bs):
            run(model, **batch.to(dev))
            for j, i in enumerate(idx):
                pos = rng.sample(range(len(seqs[i])), min(24, len(seqs[i])))
                k = store['k'][j, [pads[j] + p for p in pos]].float()
                kd = k.double()
                C[fold_of[owner[i]]] += kd.T @ kd
                n0 += k.shape[0]
        print(f'preserve L{L}: {n0} tokens from {len(seqs)} sequences', flush=True)
    finally:
        for h in hooks:
            h.remove()
    return vec, A, C, n0


def fit(vec, keys, A_tr, C_tr, version, mu, perm=None, dt=None, dev='cuda'):
    """Delta [d_model, D] fp32 on A's device. perm: key -> key whose Devanagari row (ybar, y_last) is used (control)."""
    kb = torch.stack([vec[k]['kbar'] for k in keys]).to(dev)
    kl = torch.stack([vec[k]['k_last'] for k in keys]).to(dev)
    src = [perm[k] if perm else k for k in keys]
    db = torch.stack([vec[s]['ybar'] - vec[k]['xbar'] for k, s in zip(keys, src)]).to(dev)
    dl = torch.stack([vec[s]['y_last'] - vec[k]['x_last'] for k, s in zip(keys, src)]).to(dev)
    dt = dt or (SOFT_DT[0] if version == 'soft' else PREC[0])
    kl, kb, db, dl = kl.double(), kb.double(), db.double(), dl.double()
    A = (A_tr.to(dev) + kl.T @ kl).to(dt)  # accumulated in fp64 (CPU), cast for the solve
    B = (db.T @ kb + dl.T @ kl).to(dt)  # R Omega K1^T  [d_model, D]
    C_tr = C_tr.to(dev, dt)
    D = A.shape[0]
    lam = 1e-2 * torch.trace(A) / D
    if version == 'soft':
        M = A.add_(C_tr, alpha=mu)
        del C_tr
        M.diagonal().add_(lam)
        return torch.linalg.solve(M, B.T).T.float()  # symmetric M
    P = null_projector(C_tr)
    del C_tr
    M = P @ A
    del A
    M.diagonal().add_(lam)
    return torch.linalg.solve(M, P @ B.T).T.float()  # (B P (A P + lam I)^-1)^T = (P A + lam I)^-1 P B^T


def precision_check(args, tok, model, r, L, vec, A, C, fold_of, ev_rows, apply=True):
    """fp32 vs fp64 fits on fold 0 (soft mu=10, hard); sets PREC; logs to <out>/precision_check.json and NOTES.md."""
    pid = {x['item_id']: x['pair_id'] for x in ev_rows[('hi-Latn', 0)]}
    keys = [k for k in vec if fold_of[pid[k[0]]] != 0]
    A_tr, C_tr = A.sum(0) - A[0], C.sum(0) - C[0]
    res, ok = {}, True
    for v, mu in (('soft', 10.0), ('hard', None)):
        d = {}
        for name, dt in (('fp32', torch.float32), ('fp64', torch.float64)):
            PREC[0] = dt
            d[name] = fit(vec, keys, A_tr, C_tr, v, mu, dt=dt)
        rel = float((d['fp32'].double() - d['fp64'].double()).norm() / d['fp64'].double().norm())
        au = {}
        for name in d:
            h = edit_hook(model, L, d[name], 1.0)
            try:
                for key, rs in ev_rows.items():
                    if key[0] == 'auto':
                        continue
                    sub = [x for x in rs if fold_of[x['pair_id']] == 0]
                    s = scores(tok, model, sub, r, args.batch_size)
                    au.setdefault(f'{key[0]}|{key[1]}', {})[name] = auroc(
                        [a for a, x in zip(s, sub) if x['request_type'] == 'harmful'], [a for a, x in zip(s, sub) if x['request_type'] == 'benign'])
            finally:
                h.remove()
        same = all(round(x['fp32'], 3) == round(x['fp64'], 3) for x in au.values())
        res[v + (f' mu={mu:g}' if mu else '')] = {'rel_frobenius_diff': rel, 'auroc_fold0': au, 'auroc_equal_3dp': same,
                                                   'pass': rel < 1e-3 and same}
        ok &= rel < 1e-3 and same
        del d
    PREC[0] = torch.float32 if (ok or not apply) else torch.float64
    SOFT_DT[0] = torch.float64
    out = {'layer': L, 'fold': 0, 'gamma': 1.0, 'criteria': 'rel ||D32 - D64||_F / ||D64||_F < 1e-3 and fold-0 AUROC equal to 3 dp',
           'checks': res, 'pass': ok, 'precision_used': 'fp32' if ok else 'fp64'}
    (args.out / 'precision_check.json').write_text(json.dumps(out, indent=1))
    notes = args.out.parent / 'NOTES.md'
    lines = [f"\n## Step 2 precision check ({datetime.now(timezone.utc).isoformat()}, layer {L}, fold 0, gamma 1)\n",
             'K K^T accumulated in fp64 in both arms; only the solve / eigh precision differs.\n']
    for n, c in res.items():
        worst = max(abs(x['fp32'] - x['fp64']) for x in c['auroc_fold0'].values())
        lines.append(f"- {n}: rel Delta diff {c['rel_frobenius_diff']:.2e}, max |AUROC diff| {worst:.4f}, "
                     f"equal to 3 dp: {c['auroc_equal_3dp']} -> {'pass' if c['pass'] else 'FAIL'}\n")
    lines.append(f"- Result: {'pass, fits use fp32' if ok else 'FAILED, all fits use fp64'}.\n" if apply else
                 '- Logged only (mixed mode, user decision): soft fits (all mu) use fp64; hard fits use fp32 (AUROC equal to 3 dp); the check above does not change this.\n')
    with notes.open('a') as f:
        f.writelines(lines)
    print('precision check:', json.dumps(out)[:600], flush=True)


def null_projector(C):
    ev, U = torch.linalg.eigh(C.to('cuda', PREC[0]))
    U = U[:, ev < 1e-3 * ev.max()]
    return U @ U.T


def edit_hook(model, L, delta, gamma):
    dT = delta.T.contiguous()

    def hook(_m, inputs, out):
        return out + (gamma * (inputs[0].float() @ dT)).to(out.dtype)
    return model.model.layers[L].mlp.down_proj.register_forward_hook(hook)


@torch.no_grad()
def scores(tok, model, rows, r, bs):
    store = {}
    h = stop_after(model, SCORE_LAYER - 1, store)
    out = [None] * len(rows)
    try:
        enc = [ids for ids, _ in encode(tok, rows)]
        for idx, batch, _ in batches(tok, enc, bs):
            run(model, **batch.to(model.device))
            for j, i in enumerate(idx):
                out[i] = float(store['h'][j, -1].float() @ r)
    finally:
        h.remove()
    return out


def metrics(sc, base, rows):
    """sc, base: {(script, depth): [scores]} aligned with rows[(script, depth)]."""
    res = {}
    for key, s in sc.items():
        rs = rows[key]
        pos = [x for x, r in zip(s, rs) if r['request_type'] == 'harmful']
        neg = [x for x, r in zip(s, rs) if r['request_type'] == 'benign']
        b = base[key]
        bneg = [x for x, r in zip(b, rs) if r['request_type'] == 'benign']
        bpos = [x for x, r in zip(b, rs) if r['request_type'] == 'harmful']
        res[f'{key[0]}|{key[1]}'] = {'auroc': auroc(pos, neg), 'catch': catch_rate(pos, neg),
                                     'benign_shift': sum(neg) / len(neg) - sum(bneg) / len(bneg),
                                     'harm_shift': sum(pos) / len(pos) - sum(bpos) / len(bpos),
                                     'base_gap': sum(bpos) / len(bpos) - sum(bneg) / len(bneg), 'n': len(s)}
    return res


def recovered(m, base_m, norm_m, d):
    """Fraction of the normalization gain in hi-Latn catch rate; None when normalization gains nothing."""
    gain = norm_m[f'hi-Latn|{d}']['catch'] - base_m[f'hi-Latn|{d}']['catch']
    return (m[f'hi-Latn|{d}']['catch'] - base_m[f'hi-Latn|{d}']['catch']) / gain if gain > 1e-9 else None


def eligible(m, base_m):
    for sc in ('hi-Deva', 'en'):
        for d in DEPTHS:
            x, b = m[f'{sc}|{d}'], base_m[f'{sc}|{d}']
            if b['auroc'] - x['auroc'] > 0.01 or abs(x['benign_shift']) > 0.1 * abs(b['base_gap']):
                return False
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--layers', type=int, nargs='+', default=[8, 12, 16])
    p.add_argument('--mus', type=float, nargs='+', default=[1, 10, 100])
    p.add_argument('--gammas', type=float, nargs='+', default=[0.5, 1.0])
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--limit-pairs', type=int, help='smoke test: first N correction pairs')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--precision', choices=['fp64', 'check', 'mixed'], default='mixed',
                   help='fp64: everything fp64; check: fp32 if the spot check passes, else fp64; '
                   'mixed: soft fits fp64, hard fits fp32, spot check run at the first layer and logged only')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    pairs = sorted({r['pair_id'] for r in rows_for('hi-Latn', 8)})[:args.limit_pairs]
    fold_of = folds(pairs, args.folds)
    items = {r['item_id'] for r in rows_for('hi-Latn', 8) if r['pair_id'] in fold_of}
    if args.out.exists():
        raise SystemExit(f'output exists: {args.out}')
    print(f'dry run ok: {len(pairs)} pairs, {len(items)} items, folds {args.folds}, layers {args.layers}', flush=True)
    if args.dry_run:
        return
    args.out.mkdir(parents=True)
    tok, model = load_model()
    r = r_hat(model.device)
    ev_rows = {(sc, d): [x for x in rows_for(sc, d) if x['item_id'] in items] for sc in (*EVAL, 'auto') for d in DEPTHS}

    def oof(per_fold):
        """per_fold(f, rows) -> scores; pooled out-of-fold scores per (script, depth)."""
        out = {}
        for key, rs in ev_rows.items():
            if key[0] == 'auto':
                continue
            s = [None] * len(rs)
            for f in range(args.folds):
                idx = [i for i, x in enumerate(rs) if fold_of[x['pair_id']] == f]
                for i, v in zip(idx, per_fold(f, [rs[i] for i in idx])):
                    s[i] = v
            out[key] = s
        return out
    base = {k: scores(tok, model, rs, r, args.batch_size) for k, rs in ev_rows.items()}
    norm = {(sc, d): base[('auto', d)] for sc, d in base if sc == 'hi-Latn'}
    base = {k: v for k, v in base.items() if k[0] != 'auto'}
    base_m = metrics(base, base, ev_rows)
    norm_m = metrics(norm, base, {k: ev_rows[k] for k in norm})
    results = {'base': base_m, 'normalization': norm_m, 'layers': {}}
    all_scores = {'base': base, 'normalization': norm}
    for L in args.layers:
        vec, A, C, n0 = collect(tok, model, L, items, fold_of, args.folds, args.batch_size)
        A, C = A.cpu(), C.cpu()  # fp64 fold accumulators on CPU (16 GB); per-fold sums go to the GPU
        torch.cuda.empty_cache()
        if args.precision == 'fp64':
            PREC[0] = torch.float64
        elif L == args.layers[0]:
            precision_check(args, tok, model, r, L, vec, A, C, fold_of, ev_rows, apply=args.precision == 'check')
        Pfull = null_projector(C.sum(0)).double()
        Aall = (A.sum(0) + sum(torch.outer(v['k_last'].double(), v['k_last'].double()) for v in vec.values())).cuda()
        rho = float(torch.trace(Pfull @ Aall) / torch.trace(Aall))
        null_dim = int(torch.round(torch.trace(Pfull)))
        del Pfull, Aall
        configs = [('soft', mu) for mu in args.mus] + [('hard', None)]
        deltas = {}  # (fold, version, mu) -> Delta on CPU
        pid = {x['item_id']: x['pair_id'] for x in ev_rows[('hi-Latn', 0)]}
        for f in range(args.folds):
            keys = [k for k in vec if fold_of[pid[k[0]]] != f]
            A_tr, C_tr = A.sum(0) - A[f], C.sum(0) - C[f]
            for v, mu in configs:
                deltas[(f, v, mu)] = fit(vec, keys, A_tr, C_tr, v, mu).cpu()
            del A_tr, C_tr
            print(f'L{L} fold {f} fitted', flush=True)
        lres = {'rho': rho, 'null_dim': null_dim, 'preserve_tokens': n0, 'configs': {}}

        def evaluate(dkey):
            def per_fold(f, rs):
                h = edit_hook(model, L, deltas[dkey(f)].to(model.device), dkey.gamma)
                try:
                    return scores(tok, model, rs, r, args.batch_size)
                finally:
                    h.remove()
            return per_fold
        for v, mu in configs:
            for g in args.gammas:
                name = f'{v}' + (f' mu={mu:g}' if mu is not None else '') + f' gamma={g:g}'
                dk = lambda f, v=v, mu=mu: (f, v, mu)  # noqa: E731
                dk.gamma = g
                sc = oof(evaluate(dk))
                m = metrics(sc, base, ev_rows)
                lres['configs'][name] = {'metrics': m, 'eligible': eligible(m, base_m),
                                         'recovered': {d: recovered(m, base_m, norm_m, d) for d in DEPTHS}}
                all_scores[f'L{L} {name}'] = sc
                print(f'L{L} {name}: hi-Latn catch {[round(m[f"hi-Latn|{d}"]["catch"], 3) for d in DEPTHS]} '
                      f'eligible {lres["configs"][name]["eligible"]}', flush=True)
        el = {n: c for n, c in lres['configs'].items() if c['eligible']} or lres['configs']
        best = max(el, key=lambda n: (sum(el[n]['metrics'][f'hi-Latn|{d}']['catch'] for d in DEPTHS),
                                      sum(el[n]['metrics'][f'hi-Latn|{d}']['auroc'] for d in DEPTHS)))
        lres['best'] = best
        lres['best_from_eligible'] = bool(any(c['eligible'] for c in lres['configs'].values()))
        parts = best.split()
        bv, bg = parts[0], float(parts[-1].split('=')[1])
        bmu = float(parts[1].split('=')[1]) if bv == 'soft' else None
        # shuffled-pairs control at the best fit: refit with mismatched Devanagari rows (same depth), seed 0
        rng = random.Random(0)
        for f in range(args.folds):
            keys = [k for k in vec if fold_of[pid[k[0]]] != f]
            perm = {}
            for d in DEPTHS:
                ks = [k for k in keys if k[1] == d]
                sh = ks[:]
                while len(ks) > 1 and any(a == b for a, b in zip(ks, sh)):
                    rng.shuffle(sh)
                perm.update(zip(ks, sh))
            deltas[(f, 'shuffled', bmu)] = fit(vec, keys, A.sum(0) - A[f], C.sum(0) - C[f], bv, bmu, perm).cpu()
        dk = lambda f: (f, 'shuffled', bmu)  # noqa: E731
        dk.gamma = bg
        sc = oof(evaluate(dk))
        m = metrics(sc, base, ev_rows)
        lres['control_shuffled'] = {'fit': best, 'metrics': m}
        all_scores[f'L{L} shuffled {best}'] = sc
        results['layers'][str(L)] = lres
        (args.out / 'results.json').write_text(json.dumps(results, indent=1))
        del A, C, deltas, vec
        torch.cuda.empty_cache()
    def total(L):
        c = results['layers'][L]
        return sum(c['configs'][c['best']]['metrics'][f'hi-Latn|{d}']['catch'] for d in DEPTHS)
    cand = [L for L in results['layers'] if results['layers'][L]['best_from_eligible']] or list(results['layers'])
    results['overall_best'] = {'layer': (bl := max(cand, key=total)), 'fit': results['layers'][bl]['best'],
                               'recovered': results['layers'][bl]['configs'][results['layers'][bl]['best']]['recovered']}
    results['meta'] = {'utc': datetime.now(timezone.utc).isoformat(), 'model': MODEL, 'revision': REVISION,
                       'args': {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                       'git': subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip(),
                       'versions': {k: version(k) for k in ('torch', 'transformers')}, 'gpu': torch.cuda.get_device_name(),
                       'fold_seed': 0, 'n_pairs': len(pairs), 'n_items': len(items)}
    (args.out / 'results.json').write_text(json.dumps(results, indent=1))
    torch.save({k: {f'{a}|{b}': v for (a, b), v in s.items()} for k, s in all_scores.items()}, args.out / 'scores.pt')
    with (args.out / 'prompts.jsonl').open('x', encoding='utf-8') as f:
        for (sc, d), rs in ev_rows.items():
            f.writelines(json.dumps({'script': sc, 'depth': d, 'item_id': x['item_id'], 'pair_id': x['pair_id'],
                                     'request_type': x['request_type'], 'fold': fold_of[x['pair_id']]}) + '\n' for x in rs)
    print(json.dumps(results['overall_best']), flush=True)


if __name__ == '__main__':
    main()

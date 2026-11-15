"""Causal tests with the English refusal direction (HF generation with hooks, greedy, held-out pairs only).

none:   no edit (HF baseline with the same settings, since these runs use a shorter max_new_tokens than vLLM)
ablate: remove r_hat(layer) from the residual stream after the embeddings and after every decoder layer;
        run on English single-turn harmful prompts. If refusal drops, the direction at that layer drives refusal.
add:    add alpha * r_hat(layer) to the residual stream entering decoder layer `layer` (= hidden_states[layer]) at
        every position, prompt and generated tokens; run on the chosen scripts at the chosen pool/depth, harmful AND
        benign (benign checks over-refusal).
        Scale definition: alpha = scale * ||mu_harmful - mu_benign||, the length of the English difference-in-means
        at that layer and position, fitted on correction pairs, single-turn. scale 1 = shifting by one full English
        harmful-vs-benign mean gap.
random: as add, but r_hat is a random unit vector (Gaussian, --random-seed) with the same alpha (same norm control).
gate:   decide once at the last prompt token, at the output of decoder layer `layer`-1:
          g = max(0, h_last . r_hat - c[script])   (c = benign 95th percentile, calib.py gate.json)
        then add beta * g * r_hat at the last prompt token and at every generated token (0 if the gate did not fire).
gate_random: the same g (decided on r_hat) and beta, but adds beta * g * v, v a random unit vector (--random-seed):
        same prompts edited, same edit norm, different direction.
Output rows are compatible with codeswitch/guard_judge.py (Qwen3Guard response moderation). Never overwrites.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random

import torch

from direction import project_out
from extract import POSITIONS, encode, limit, load_model
from prompts import build


def select(mode, args):
    if mode in ('none', 'ablate') and args.target == 'english':
        rows = [r for r in build('single') if r['switch'] == 'en' and r['request_type'] == 'harmful' and r['split'] == 'heldout']
    elif args.target == 'single':  # one-turn prompts (system + final request), same pairs and scripts as the depth target
        rows = [r for r in build('single') if r['split'] == args.split and r['switch'] in args.scripts]
    else:
        rows = [r for r in build('depth', args.split) if r['pool'] == args.pool and r['depth'] == args.depth and r['switch'] in args.scripts]
    if args.sample_pairs:
        pairs = sorted({r['pair_id'] for r in rows})
        keep = set(random.Random(args.sample_seed).sample(pairs, args.sample_pairs))
        rows = [r for r in rows if r['pair_id'] in keep]
    return limit(rows, args.limit_pairs) if args.limit_pairs else rows


def hooks(model, mode, r_hat, layer, alpha):
    layers = model.model.layers

    def edit(fn):
        def hook(_module, _inputs, output):
            if isinstance(output, tuple):
                return (fn(output[0]),) + tuple(output[1:])
            return fn(output)
        return hook
    if mode == 'ablate':
        targets = [model.model.embed_tokens, *layers]
        return [m.register_forward_hook(edit(lambda h: project_out(h, r_hat))) for m in targets]
    if mode in ('add', 'random'):  # hidden_states[layer] is the output of embeddings (layer 0) or of decoder layer layer-1
        target = model.model.embed_tokens if layer == 0 else layers[layer - 1]
        return [target.register_forward_hook(edit(lambda h: h + alpha * r_hat.to(h.dtype)))]
    return []


def gate_hook(model, layer, r_gate, r_add, c, beta, state, push_steps=None, span=False):
    """Decide-once gate. A call with more than one position (prefill / teacher forcing) decides g at position
    state['start'] (default: the last position) and edits positions >= start; single-position calls (decoding) reuse g.
    The fired g per batch row is left in state['g']. A per-row threshold tensor in state['c'] overrides the scalar c.
    push_steps (A1/A2): None = edit the decision position and every later position (default); k = edit the decision position
    and only the next k positions (generated tokens when decoding, answer tokens when teacher forcing); 0 = prefill only.
    span (A1b): edit the whole final user turn in the prompt, positions state['span_start'][row] .. decision position
    (per row, left padding included in span_start), and no generated / answer token."""
    def hook(_module, _inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        if span:
            if h.shape[1] == 1 and 'g' in state:  # decoding: A1b never edits generated tokens
                return output
            s = state.get('start', h.shape[1] - 1)
            state['g'] = (h[:, s].float() @ r_gate - state.get('c', c)).clamp(min=0)
            idx = torch.arange(h.shape[1], device=h.device)
            mask = (idx[None, :] >= state['span_start'].to(h.device)[:, None]) & (idx[None, :] <= s)  # [b, T]
            delta = (beta * state['g'])[:, None, None] * mask[..., None].float() * r_add
            h = h + delta.to(h.dtype)
            return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h
        if h.shape[1] > 1 or 'g' not in state:
            s = state.get('start', h.shape[1] - 1)
            state['g'] = (h[:, s].float() @ r_gate - state.get('c', c)).clamp(min=0)
            state['step'] = 0
            end = h.shape[1] if push_steps is None else min(h.shape[1], s + 1 + push_steps)
        else:
            state['step'] += 1
            s = 0
            if push_steps is not None and state['step'] > push_steps:
                return output
            end = h.shape[1]
        delta = (beta * state['g'])[:, None, None] * r_add
        h = torch.cat([h[:, :s], h[:, s:end] + delta.to(h.dtype), h[:, end:]], 1)
        return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h
    return [model.model.layers[layer - 1].register_forward_hook(hook)]


def output_path(args, layer):
    tag = (f'none-{args.target}' + (f'-{args.split}' if args.target in ('depth', 'single') else '') if args.mode == 'none'
           else f'{args.mode}-L{layer}-{args.position}' + (f'-x{args.scale:g}' if args.mode in ('add', 'random') else '')
           + (f'-b{args.beta:g}' if args.mode in ('gate', 'gate_random') else '')
           + (f'-k{args.push_steps}' if args.push_steps is not None else '')
           + ('-span' if args.push_span else '')
           + ('-fb' if args.full_batch else '')) \
        + (f'-{args.final_message}' if args.final_message else '')
    return args.out / f'{tag}.jsonl'


def final_turn_start(tok, ids):
    """Index of the <|START_OF_TURN_TOKEN|> that opens the LAST user turn (followed by <|USER_TOKEN|>)."""
    start, user = tok.convert_tokens_to_ids('<|START_OF_TURN_TOKEN|>'), tok.convert_tokens_to_ids('<|USER_TOKEN|>')
    return max(i for i in range(len(ids) - 1) if ids[i] == start and ids[i + 1] == user)


def dry_run(args):
    """Fail now, not hours into a queue: argument combinations, input files, prompt count, output not yet present."""
    gate_mode = args.mode in ('gate', 'gate_random')
    if gate_mode and (args.beta is None or not (args.gate_file or args.tau_file)):
        raise SystemExit('gate modes need --beta and --gate-file or --tau-file')
    if args.full_batch and (args.unfired_from or not args.gate_file):
        raise SystemExit('--full-batch needs --gate-file and excludes --unfired-from')
    if (args.push_steps is not None or args.push_span) and not gate_mode:
        raise SystemExit('--push-steps / --push-span are for gate modes')
    if args.push_steps is not None and args.push_span:
        raise SystemExit('--push-steps and --push-span are exclusive')
    if args.orth_random and args.mode != 'gate_random':
        raise SystemExit('--orth-random is for gate_random')
    if args.final_message == 'auto' and not args.xlit:
        raise SystemExit('--final-message auto needs --xlit')
    for f in (args.directions, args.gate_file, args.tau_file, args.xlit):
        if f is not None and not f.exists():
            raise SystemExit(f'missing input {f}')
    rows = select(args.mode, args)
    if not rows:
        raise SystemExit('empty prompt set')
    for layer in ([None] if args.mode == 'none' else args.layers):
        if output_path(args, layer).exists():
            raise SystemExit(f'output exists: {output_path(args, layer)}')
    print(f'dry run ok: mode {args.mode}, {len(rows)} prompts, out {[str(output_path(args, la)) for la in ([None] if args.mode == "none" else args.layers)]}', flush=True)


def final_message(tok, rows, kind, xlit=None):
    """Replace the final user message the model reads for hi-Latn rows (in place); return the batch order fixed by the
    ROMANIZED prompts, so every --final-message variant has the same batch membership. 'request' stays romanized."""
    assert {r['switch'] for r in rows} == {'hi-Latn'}, 'normalization runs are romanized-only'
    encoded = encode(tok, rows)
    order = sorted(range(len(rows)), key=lambda i: -len(encoded[i][0]))
    if kind == 'auto':
        text = {x['item_id']: x['auto'] for x in map(json.loads, xlit.read_text(encoding='utf-8').splitlines())}
    elif kind == 'gold':
        deva = [r for r in build('depth', rows[0]['split']) if r['switch'] == 'hi-Deva']
        text = {r['item_id']: r['request'] for r in deva}
    for r in rows:
        assert r['messages'][-1] == {'role': 'user', 'content': r['request']}
        r['model_request'] = r['request'] if kind == 'romanized' else text[r['item_id']]
        r['final_message'] = kind
        r['messages'] = r['messages'][:-1] + [{'role': 'user', 'content': r['model_request']}]
    return order


@torch.no_grad()
def generate(tok, model, rows, batch_size, max_new_tokens, state=None, order=None):
    """order: fixed row order (hence fixed batch membership); default = by decreasing prompt length.
    Rows carrying 'gate_c' give the gate a per-row threshold; 'gate_fold' a per-row direction index."""
    encoded = encode(tok, rows)
    order = order if order is not None else sorted(range(len(rows)), key=lambda i: -len(encoded[i][0]))
    out = [None] * len(rows)
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        batch = tok.pad({'input_ids': [encoded[i][0] for i in idx]}, return_tensors='pt').to(model.device)
        if state is not None:
            state.pop('g', None)
            if 'gate_c' in rows[idx[0]]:
                state['c'] = torch.tensor([rows[i]['gate_c'] for i in idx], device=model.device)
            if 'gate_fold' in rows[idx[0]]:  # per-row detection direction (exploratory OOF gate)
                state['fold'] = torch.tensor([rows[i]['gate_fold'] for i in idx], device=model.device)
            width = batch['input_ids'].shape[1]  # left padding: row position = pad + index in its own ids
            state['span_start'] = torch.tensor([width - len(encoded[i][0]) + final_turn_start(tok, encoded[i][0]) for i in idx])
        gen = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
        for j, i in enumerate(idx):
            if state is not None:
                rows[i]['gate_g'] = float(state['g'][j])
            new = gen[j, batch['input_ids'].shape[1]:].tolist()
            stopped = tok.eos_token_id in new
            new = new[:new.index(tok.eos_token_id)] if stopped else new
            out[i] = (tok.decode(new, skip_special_tokens=True), 'stop' if stopped else 'length', len(new))
        print(f'{min(start + batch_size, len(order))}/{len(order)}', flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['none', 'ablate', 'add', 'random', 'gate', 'gate_random'], required=True)
    parser.add_argument('--beta', type=float, help='gate modes')
    parser.add_argument('--push-steps', type=int, help='gate modes: edit only the decision token + the next k tokens '
                        '(0 = prefill only, A1; k > 0 = A2); default: every generated token')
    parser.add_argument('--dry-run', action='store_true', help='check arguments, input files, the prompt set and that the output '
                        'file does not exist yet, then exit')
    parser.add_argument('--push-span', action='store_true', help='gate (A1b): push on the whole final user turn in the prompt '
                        '(from its <|START_OF_TURN_TOKEN|> through the last prompt token), no generated token')
    parser.add_argument('--orth-random', action='store_true', help='gate_random: orthogonalize the random vector against r_hat')
    parser.add_argument('--gate-file', type=Path, help='calib.py gate.json (gate modes)')
    parser.add_argument('--unfired-from', type=Path, help='gate modes: unedited run with the same prompts and settings; '
                        'rows whose gate does not fire copy that reply instead of regenerating it')
    parser.add_argument('--final-message', choices=['romanized', 'auto', 'gold'],
                        help='hi-Latn rows: what the MODEL reads as the final user message (prefix untouched); the row '
                        "'request' stays romanized for the judge. Also fixes batch order by romanized prompt length.")
    parser.add_argument('--xlit', type=Path, help='normalize xlit.jsonl (for --final-message auto)')
    parser.add_argument('--tau-file', type=Path, help='normalize tau.json: per-row gate threshold, no copying')
    parser.add_argument('--full-batch', action='store_true', help='gate with --gate-file: per-row c, all prompts generated in '
                        'the same batches as the unedited baseline, nothing copied (rule 0.2)')
    parser.add_argument('--target', choices=['english', 'depth', 'single'], default='english',
                        help="none/ablate prompt set ('english'); 'depth' = pool/depth conversations; 'single' = one-turn prompts")
    parser.add_argument('--directions', type=Path, required=True)
    parser.add_argument('--layers', type=int, nargs='+', default=[0], help='one output file per layer (ignored for none)')
    parser.add_argument('--position', choices=POSITIONS, default='last', help='which fitted direction to use')
    parser.add_argument('--scale', type=float, default=1.0)
    parser.add_argument('--pool', default='v2')
    parser.add_argument('--depth', type=int, default=8)
    parser.add_argument('--scripts', nargs='+', default=['hi-Latn', 'en-hi'])
    parser.add_argument('--split', choices=['heldout', 'correction'], default='heldout', help='depth and single targets')
    parser.add_argument('--sample-pairs', type=int, help='random sample of pairs (depth target)')
    parser.add_argument('--sample-seed', type=int, default=42)
    parser.add_argument('--random-seed', type=int, default=0)
    parser.add_argument('--limit-pairs', type=int)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--max-new-tokens', type=int, default=512)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.dry_run:
        dry_run(args)
        return
    dirs = torch.load(args.directions)
    p = POSITIONS.index(args.position)
    rows = select(args.mode, args)
    args.out.mkdir(parents=True, exist_ok=True)
    tok, model = load_model()
    order = None
    if args.final_message:
        order = final_message(tok, rows, args.final_message, args.xlit)
    for layer in ([None] if args.mode == 'none' else args.layers):
        path = output_path(args, layer)
        if path.exists():
            raise FileExistsError(path)
        r_hat = alpha = None
        if layer is not None:
            r_hat = dirs['r_hat']['en'][layer, p].to(model.device)
            if args.mode in ('random', 'gate_random'):
                g = torch.Generator().manual_seed(args.random_seed)
                v = torch.randn(r_hat.shape[0], generator=g)
                if args.orth_random:  # u <- u - (r_hat . u) r_hat, then normalize
                    r_en = dirs['r_hat']['en'][layer, p].float()
                    v = v - (v @ r_en) * r_en
                r_hat = (v / v.norm()).to(model.device)
            alpha = float(dirs['norm']['en'][layer, p]) * args.scale
        gate = None
        if args.tau_file or args.full_batch:
            # per-row threshold, every row generated in the baseline batch order (fixed order for --final-message runs,
            # else the default order = the unedited baseline's), nothing copied
            assert args.mode in ('gate', 'gate_random') and not args.unfired_from
            if args.tau_file:
                tau = json.loads(args.tau_file.read_text())
                assert tau['layer'] == layer
                gate = {'c_r': tau['tau']}
            else:
                gate = json.loads(args.gate_file.read_text())
                assert gate['layer'] == layer and gate['random_seed'] == args.random_seed
            for r in rows:
                r['gate_c'] = gate['c_r'][r['switch']]
            state = {}
            handles = gate_hook(model, layer, dirs['r_hat']['en'][layer, p].float().to(model.device), r_hat.float(), 0.0,
                                args.beta, state, args.push_steps, args.push_span)
            try:
                results = generate(tok, model, rows, args.batch_size, args.max_new_tokens, state, order)
            finally:
                for h in handles:
                    h.remove()
        elif args.mode in ('gate', 'gate_random'):
            gate = json.loads(args.gate_file.read_text())
            assert gate['layer'] == layer and gate['random_seed'] == args.random_seed, 'gate.json does not match --layers/--random-seed'
            r_gate = dirs['r_hat']['en'][layer, p].float().to(model.device)
            results = [None] * len(rows)
            unfired = {}
            if args.unfired_from:  # g = 0 -> the edit is exactly zero -> reuse the unedited reply (same prompts/settings)
                unfired = {b['conversation_id']: b for b in map(json.loads, args.unfired_from.read_text().splitlines())}
                assert unfired[rows[0]['key']]['generation']['max_new_tokens'] == args.max_new_tokens
            for s in sorted({r['switch'] for r in rows}):
                idx = [i for i, r in enumerate(rows) if r['switch'] == s]
                state = {}
                handles = gate_hook(model, layer, r_gate, r_hat.float(), gate['c_r'][s], args.beta, state, args.push_steps, args.push_span)
                try:
                    if unfired:
                        generate(tok, model, [rows[i] for i in idx], args.batch_size, 1, state)  # decide g only
                        for i in idx:
                            rows[i]['gate_g_decision'] = rows[i]['gate_g']
                            if rows[i]['gate_g'] == 0:
                                b = unfired[rows[i]['key']]
                                results[i] = (b['response'], b['finish_reason'], b['generated_tokens'])
                                rows[i]['copied_from_unedited'] = True
                        idx = [i for i in idx if results[i] is None]
                    for i, res in zip(idx, generate(tok, model, [rows[i] for i in idx], args.batch_size, args.max_new_tokens, state)):
                        results[i] = res
                        rows[i]['copied_from_unedited'] = False
                finally:
                    for h in handles:
                        h.remove()
        else:
            handles = hooks(model, args.mode, r_hat, layer, alpha)
            try:
                results = generate(tok, model, rows, args.batch_size, args.max_new_tokens, order=order)
            finally:
                for h in handles:
                    h.remove()
        edit = {'mode': args.mode, 'layer': layer, 'position': args.position, 'alpha': alpha, 'scale': args.scale,
                'beta': args.beta if gate else None, 'c': gate['c_r'] if gate else None, 'push_steps': args.push_steps,
                'push_span': args.push_span, 'random_orthogonal_to_r_hat': bool(args.orth_random),
                'full_batch': bool(args.full_batch or args.tau_file), 'copied_unfired': bool(args.unfired_from),
                'gate_file': str(args.gate_file or args.tau_file) if gate else None,
                'final_message': args.final_message, 'xlit_file': str(args.xlit) if args.xlit else None,
                'batch_order': 'fixed: decreasing romanized prompt length' if order is not None else 'decreasing prompt length',
                'direction': 'random' if args.mode in ('random', 'gate_random') else 'en',
                'random_seed': args.random_seed if args.mode in ('random', 'gate_random') else None,
                'split': args.split, 'sample_pairs': args.sample_pairs, 'sample_seed': args.sample_seed,
                'scale_definition': 'alpha = scale * ||mu_harmful - mu_benign|| (English, correction pairs, single-turn, same layer/position)',
                'directions_file': str(args.directions)}
        gen = {'backend': 'hf', 'do_sample': False, 'max_new_tokens': args.max_new_tokens}
        with path.open('x', encoding='utf-8') as f:
            for r, (text, finish, n) in zip(rows, results):
                f.write(json.dumps({**{k: v for k, v in r.items() if k != 'messages'}, 'conversation_id': r['key'],
                                    'response': text, 'finish_reason': finish, 'generated_tokens': n, 'edit': edit,
                                    'generation': gen, 'utc': datetime.now(timezone.utc).isoformat()},
                                   ensure_ascii=False) + '\n')
        print('saved', path, flush=True)


if __name__ == '__main__':
    main()

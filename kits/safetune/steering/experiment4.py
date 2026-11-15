#!/usr/bin/env python3
"""Frozen-gate diagnostics: offline plots and unsteered, fit-only layer extraction."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
import random
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cast_experiment import load_jsonl, load_policies, render, sha256, policy_calibration

MODES = ('policy', 'caa', 'cast')


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')


def margin(score, threshold, comparator):
    if comparator == 'smaller':
        return score - threshold
    if comparator == 'larger':
        return threshold - score
    raise ValueError(f'Unknown comparator: {comparator}')


def saved_gates(output, conditions):
    gates = output['policy_gates']
    if set(gates) != set(output['active_policies']):
        raise ValueError('Recorded gates differ from active policies')
    result = {}
    for name, gate in gates.items():
        condition = conditions[name]
        if any(gate[k] != condition[ck] for k, ck in
               [('layer', 'layer'), ('threshold', 'threshold'), ('comparator', 'comparator')]):
            raise ValueError('Saved gate differs from frozen calibration')
        score, fired = gate['scores'][0], gate['fires'][0]
        m = margin(score, gate['threshold'], gate['comparator'])
        if fired != (m > 0):
            raise ValueError('Gate decision and margin disagree')
        result[name] = {'score': score, 'margin': m, 'fired': fired, **{
            k: gate[k] for k in ('layer', 'threshold', 'comparator')}}
    if output['gate_fired'] != any(g['fired'] for g in result.values()):
        raise ValueError('Recorded OR gate disagrees with per-policy decisions')
    return result


def switch_pairs(rows):
    """Pair only exact request/history matches with different active policies."""
    groups = defaultdict(list)
    for row in rows:
        key = (row['family_id'], row['language'], row['user_request'],
               json.dumps(row.get('prior_messages', []), sort_keys=True, ensure_ascii=False))
        groups[key].append(row)
    pairs, excluded = [], []
    for key, group in groups.items():
        allowed = [r for r in group if r['expected_action'] == 'allow']
        refused = [r for r in group if r['expected_action'] == 'refuse']
        if len(allowed) == len(refused) == 1 and allowed[0]['active_policies'] != refused[0]['active_policies']:
            pairs.append((allowed[0], refused[0]))
        elif len(group) > 1:
            excluded.append({'family_id': key[0], 'ids': [r['sample_id'] for r in group],
                             'reason': 'not exactly one allow and one refuse with different policies'})
    return pairs, excluded


def benefit_category(policy, caa):
    if not isinstance(policy, bool) or not isinstance(caa, bool):
        return None
    return {(False, True): 'Steering helps', (True, False): 'Steering harms',
            (True, True): 'Both pass', (False, False): 'Both fail'}[(policy, caa)]


def validate_inputs(args):
    import torch
    rows = [r for r in load_jsonl(args.data) if r['split'] == 'validation']
    settings = json.loads(args.outputs.with_suffix('.settings.json').read_text())
    cache = torch.load(args.cache, map_location='cpu', weights_only=True)
    meta = cache['metadata']
    for key, expected in [('dataset_sha256', sha256(args.data)), ('policies_sha256', sha256(args.policies))]:
        if settings[key] != expected or meta[key] != expected:
            raise ValueError(f'{key} mismatch')
    for key in ('model', 'model_revision', 'dtype', 'calib_per_class', 'seed',
                'condition_layers', 'behavior_layers'):
        if settings[key] != meta[key]:
            raise ValueError(f'Calibration/generation {key} mismatch')
    outputs = {}
    for output in read(args.outputs):
        key = (output['sample_id'], output['mode'])
        if key in outputs:
            raise ValueError(f'Duplicate output {key}')
        outputs[key] = output
    if any((r['sample_id'], m) not in outputs for r in rows for m in MODES):
        raise ValueError('Incomplete validation outputs; no silently selected subset')
    judgments = {}
    for verdict in read(args.judgments):
        key = (verdict['sample_id'], verdict['mode'])
        # Match scorer resume semantics: later retries replace previous errors.
        judgments[key] = verdict
    return rows, settings, cache, outputs, judgments


def diagnostics(args):
    rows, settings, cache, outputs, judgments = validate_inputs(args)
    args.out.mkdir(parents=True, exist_ok=True)
    records = []
    categories = {name: Counter({'ON': 0, 'OFF': 0}) for name in
                  ('Steering helps', 'Steering harms', 'Both pass', 'Both fail')}
    missing = Counter()
    text_check = Counter()
    for row in rows:
        sid = row['sample_id']
        cast_output = outputs[sid, 'cast']
        gates = saved_gates(cast_output, cache['conditions'])
        record = {k: row[k] for k in ('sample_id', 'family_id', 'language', 'expected_action', 'active_policies')}
        record.update({'policy_gates': gates, 'gate_margin': max(g['margin'] for g in gates.values()),
                       'gate_fired': cast_output['gate_fired']})
        for mode in MODES:
            verdict = judgments.get((sid, mode), {})
            valid = not verdict.get('error') and verdict.get('observed_action') in ('allow', 'refuse')
            record[mode+'_pass'] = verdict.get('success') if valid else None
            record[mode+'_action'] = verdict.get('observed_action') if valid else None
            record[mode+'_status'] = ('missing_or_error' if not valid else
                                      'uncertain' if verdict.get('success') is None else 'scored')
            if record[mode+'_status'] != 'scored':
                missing[mode+':'+record[mode+'_status']] += 1
        if all(isinstance(record[m+'_pass'], bool) for m in MODES):
            category = benefit_category(record['policy_pass'], record['caa_pass'])
            categories[category]['ON' if record['gate_fired'] else 'OFF'] += 1
        chosen = 'caa' if record['gate_fired'] else 'policy'
        same = cast_output['generated_text'] == outputs[sid, chosen]['generated_text']
        record['selected_branch'] = chosen
        record['selected_branch_text_equal'] = same
        text_check['equal' if same else 'different'] += 1
        records.append(record)
    by_id = {r['sample_id']: r for r in records}
    pairs, excluded = switch_pairs(rows)
    paired = []
    for allowed, refused in pairs:
        a, r = by_id[allowed['sample_id']], by_id[refused['sample_id']]
        quadrant = { (False, True): 'Correct gating', (True, True): 'Both ON',
                     (False, False): 'Both OFF', (True, False): 'Reversed'}[(a['gate_fired'], r['gate_fired'])]
        paired.append({'family_id': a['family_id'], 'language': a['language'],
            'allowed_id': a['sample_id'], 'restricted_id': r['sample_id'],
            'allowed_margin': a['gate_margin'], 'restricted_margin': r['gate_margin'],
            'quadrant': quadrant})
    with (args.out/'figure4_diagnostics.jsonl').open('w') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False)+'\n')
    write(args.out/'policy_switch_pairs.json', paired)
    clusters = defaultdict(list)
    for pair in paired:
        clusters[pair['family_id']].append(pair['quadrant'] == 'Correct gating')
    rng = random.Random(42)
    draws = []
    for _ in range(2000):
        sampled = rng.choices(list(clusters.values()), k=len(clusters))
        outcomes = [x for group in sampled for x in group]
        if outcomes:
            draws.append(sum(outcomes)/len(outcomes))
    draws.sort()
    summary = {'pair_unique_families': len(clusters),
        'pair_family_bootstrap_95ci': [draws[49], draws[1949]] if draws else None,
        'n_validation_binary': len(rows), 'n_nonbinary_excluded': sum(
        r['split']=='validation' and r['expected_action'] not in ('allow','refuse')
        for r in load_jsonl(args.data, binary_only=False)),
        'pairs': len(paired), 'pair_quadrants': dict(Counter(p['quadrant'] for p in paired)),
        'excluded_pair_groups': excluded, 'benefit_categories': categories,
        'missing_or_uncertain': dict(missing), 'branch_decoded_text_comparison': dict(text_check),
        'token_identity_check': 'unavailable: original outputs did not save token IDs',
        'by_policy': {}, 'by_language': {}}
    for key, field in [('by_policy', 'active_policies'), ('by_language', 'language')]:
        for value in sorted({'+'.join(r[field]) if isinstance(r[field], list) else r[field] for r in records}):
            group = [r for r in records if ('+'.join(r[field]) if isinstance(r[field], list) else r[field])==value]
            summary[key][value] = {'n': len(group), 'gate_on': sum(r['gate_fired'] for r in group)}
    write(args.out/'diagnostic_summary.json', summary)
    write(args.out/'input_manifest.json', {'settings': settings, 'calibration_metadata': cache['metadata'],
        'input_sha256': {str(p): sha256(p) for p in (args.data, args.policies, args.cache, args.outputs, args.judgments)},
        'source_sha256': sha256(Path(__file__)), 'layer_indices': 'zero-based decoder blocks'})
    plot_saved(args, records, paired, categories, cache)
    print(json.dumps(summary, indent=2))


def savefig(fig, out, name):
    for suffix in ('png', 'pdf'):
        fig.savefig(out/f'{name}.{suffix}', dpi=200, bbox_inches='tight')
    import matplotlib.pyplot as plt
    plt.close(fig)


def plot_saved(args, records, pairs, categories, cache):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(cache['conditions']), figsize=(14, 4), squeeze=False)
    for ax, (name, c) in zip(axes[0], cache['conditions'].items()):
        # Only single-policy cases carry an unambiguous label for this gate.
        groups = [[r['policy_gates'][name]['score'] for r in records
                   if r['active_policies']==[name] and r['expected_action']==label]
                  for label in ('allow', 'refuse')]
        ax.boxplot(groups, tick_labels=['Allowed', 'Restricted'], showfliers=True)
        ax.axhline(c['threshold'], color='gray', ls='--')
        ax.set_title(f"{name}\nlayer {c['layer']}; n={list(map(len,groups))}", fontsize=9)
        ax.set_ylabel('CAST condition similarity')
    savefig(fig, args.out, 'A_selected_gate_scores')
    fig, ax = plt.subplots(figsize=(8, 7))
    for p in pairs:
        ax.scatter(p['allowed_margin'], p['restricted_margin'])
        ax.annotate(p['restricted_id'], (p['allowed_margin'], p['restricted_margin']), fontsize=6)
    ax.axhline(0, color='gray', ls='--'); ax.axvline(0, color='gray', ls='--')
    ax.set(xlabel='Allowed-policy gate margin (OR maximum)', ylabel='Restricted-policy gate margin (OR maximum)',
           title=f'Policy switching: {len(pairs)} exact binary pairs')
    savefig(fig, args.out, 'B_policy_switch_margins')
    fig, ax = plt.subplots(figsize=(9, 5))
    names = list(categories)
    for offset, label in [(-.18, 'ON'), (.18, 'OFF')]:
        bars = ax.bar([i+offset for i in range(len(names))], [categories[n][label] for n in names], width=.36, label='Gate '+label)
        ax.bar_label(bars, padding=2)
    ax.set_xticks(range(len(names)), names); ax.set_ylabel('Examples scored in all three modes')
    ax.legend(); savefig(fig, args.out, 'D_steering_benefit')


def extract(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from safetune._refusal_helpers import _get_decoder_layers
    from safetune.steer import fit_cast_condition
    from safetune.interventions.steer.cast import _cast_similarity
    from cast_experiment import CalibTokenizer
    rows, settings, cache, _, _ = validate_inputs(args)
    if not 0 < args.gpu_memory_fraction <= 1:
        raise ValueError('GPU memory fraction must be in (0, 1]')
    torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, 0)
    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out/'activations.pt'
    if target.exists():
        raise FileExistsError(target)
    from datetime import datetime, timezone
    from importlib.metadata import version
    write(args.out/'extraction_environment.json', {
        'recorded_utc': datetime.now(timezone.utc).isoformat(),
        'gpu': torch.cuda.get_device_name(0), 'memory_fraction': args.gpu_memory_fraction,
        'cuda': torch.version.cuda, 'versions': {name:version(name) for name in
            ('torch','transformers','safetune','scikit-learn','matplotlib')}})
    tok = AutoTokenizer.from_pretrained(settings['model'], revision=settings['model_revision'])
    if tok.get_chat_template() != cache['metadata']['chat_template']:
        raise ValueError('Chat template mismatch')
    model = AutoModelForCausalLM.from_pretrained(settings['model'], revision=settings['model_revision'],
        dtype=torch.bfloat16 if settings['dtype']=='bf16' else torch.float16, device_map={'': 'cuda:0'}).eval()
    model.requires_grad_(False)
    decoder = _get_decoder_layers(model)
    selected = sorted(set(args.layers + [c['layer'] for c in cache['conditions'].values()] + [args.embedding_layer]))
    if any(not 0 <= i < len(decoder) for i in selected):
        raise ValueError('Invalid layer index')
    by_policy = policy_calibration(load_jsonl(args.data), settings['calib_per_class'], settings['seed'])
    specs = load_policies(args.policies)
    expected_ids = {name: [r['sample_id'] for r in refuse+allow] for name,(refuse,allow) in by_policy.items()}
    if expected_ids != cache['metadata']['policy_sample_ids']:
        raise ValueError('Calibration sample IDs differ from frozen cache')
    prompts = [render(tok, r, specs) for pair in by_policy.values() for r in pair[0]+pair[1]]
    adapter = CalibTokenizer(tok, prompts)
    layer_conditions = {}
    for name, (refuse, allow) in by_policy.items():
        layer_conditions[name] = {}
        for layer in args.layers:
            # Preserve the actual selected vector/threshold exactly; fit-only auxiliary vectors elsewhere.
            if layer == cache['conditions'][name]['layer']:
                c = cache['conditions'][name]
            else:
                fitted = fit_cast_condition(model, [render(tok,r,specs) for r in refuse],
                    [render(tok,r,specs) for r in allow], adapter, candidate_layers=[layer],
                    pool=cache['conditions'][name]['pool'], chat_template=True)
                c = {'vector': fitted.condition_vector.cpu(), 'layer': fitted.condition_layer,
                     'threshold': fitted.threshold, 'comparator': fitted.comparator, 'pool': fitted.pool,
                     'fit_f1': fitted.f1}
            layer_conditions[name][layer] = c
    captured = {}
    def hook(layer):
        def capture(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            # Single unpadded prompt, with the saved CAST pooling.
            captured[layer] = h[0].detach().float().cpu()
        return capture
    handles = [decoder[layer].register_forward_hook(hook(layer)) for layer in selected]
    activations = {layer: [] for layer in selected}
    try:
        for i, row in enumerate(rows):
            captured.clear()
            enc = tok(render(tok,row,specs), return_tensors='pt', add_special_tokens=False).to(model.device)
            if enc['input_ids'].shape[1] > settings['max_input_tokens']:
                raise ValueError('Input exceeds frozen input limit')
            with torch.inference_mode():
                model(**enc, use_cache=False)
            for layer in selected:
                # Cache both pooling variants; per-policy scores use that policy's saved pool.
                activations[layer].append({'last': captured[layer][-1].clone(), 'mean': captured[layer].mean(0)})
            print(f'Activations {i+1}/{len(rows)}', flush=True)
    finally:
        for handle in handles:
            handle.remove()
    scores = []
    for i,row in enumerate(rows):
        scores.append({'sample_id': row['sample_id'], 'active_policy_scores': {
            name: {layer: _cast_similarity(activations[layer][i][c['pool']], c['vector'])
                   for layer,c in layer_conditions[name].items()} for name in row['active_policies']}})
    torch.save({'sample_ids':[r['sample_id'] for r in rows], 'activations':activations,
        'layer_conditions':layer_conditions, 'scores':scores, 'embedding_layer':args.embedding_layer,
        'embedding_pool':'last', 'cache_sha256':sha256(args.cache), 'source_sha256':sha256(Path(__file__)),
        'model':settings['model'], 'model_revision':settings['model_revision'],
        'gpu_memory_fraction':args.gpu_memory_fraction,
        'note':'Unsteered policy-prefill activations; auxiliary vectors fit only on original FIT examples'}, target)
    write(args.out/'layerwise_scores.json', scores)


def refusal_surface(coordinates, actions, grid, bandwidth):
    """Gaussian-smoothed observed actions; unknown labels contribute no weight."""
    import numpy as np
    xy = np.asarray(coordinates, dtype=float)
    points = np.asarray(grid, dtype=float)
    if bandwidth <= 0:
        raise ValueError('Surface bandwidth must be positive')
    valid = np.array([a in ('allow', 'refuse') for a in actions])
    if not valid.any():
        return np.full(len(points), np.nan)
    values = np.array([a == 'refuse' for a in actions], dtype=float)[valid]
    distances = ((points[:, None, :] - xy[valid][None, :, :]) ** 2).sum(-1)
    # Subtract a row constant to avoid underflow without changing weighted means.
    distances -= distances.min(axis=1, keepdims=True)
    weights = np.exp(-distances / (2 * bandwidth ** 2))
    return (weights @ values) / weights.sum(axis=1)


def plot_activations(args):
    import torch
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    rows, settings, cache, outputs, _ = validate_inputs(args)
    saved = torch.load(args.out/'activations.pt', map_location='cpu', weights_only=True)
    if saved['sample_ids'] != [r['sample_id'] for r in rows] or saved['cache_sha256'] != sha256(args.cache):
        raise ValueError('Activation identity mismatch')
    scores = {r['sample_id']:r for r in saved['scores']}
    from safetune.interventions.steer.cast import _cast_similarity, _condition_met
    errors = []
    mismatches = 0
    for i, sid in enumerate(saved['sample_ids']):
        for name, gate in outputs[sid, 'cast']['policy_gates'].items():
            c = cache['conditions'][name]
            h = saved['activations'][c['layer']][i][c['pool']]
            score = _cast_similarity(h, c['vector'])
            errors.append(abs(score - gate['scores'][0]))
            mismatches += _condition_met(score, c['threshold'], c['comparator']) != gate['fires'][0]
    write(args.out/'activation_parity.json', {
        'n_active_gate_comparisons': len(errors), 'max_abs_score_difference': max(errors),
        'mean_abs_score_difference': sum(errors)/len(errors), 'gate_decision_mismatches': mismatches,
        'note': 'Single-prompt extraction versus original batched prefill; frozen vectors/thresholds.'})
    stats = {}
    for name, conditions in saved['layer_conditions'].items():
        stats[name] = {}
        for layer in conditions:
            stats[name][layer] = {}
            for label in ('allow', 'refuse'):
                values = [scores[r['sample_id']]['active_policy_scores'][name][layer]
                          for r in rows if r['active_policies']==[name] and r['expected_action']==label]
                stats[name][layer][label] = {'n': len(values),
                    'q25_median_q75': np.quantile(values, [.25,.5,.75]).tolist()}
    write(args.out/'layerwise_statistics.json', stats)
    fig, axes = plt.subplots(len(cache['conditions']),1,figsize=(9,11))
    for ax, name in zip(axes,cache['conditions']):
        layers = sorted(saved['layer_conditions'][name])
        for offset,label,color in [(-.17,'allow','#3977b8'),(.17,'refuse','#cc735c')]:
            groups = [[scores[r['sample_id']]['active_policy_scores'][name][layer]
                for r in rows if r['active_policies']==[name] and r['expected_action']==label] for layer in layers]
            boxes=ax.boxplot(groups,positions=np.arange(len(layers))+offset,widths=.28,patch_artist=True)
            for box in boxes['boxes']:box.set_facecolor(color)
        ax.set_xticks(range(len(layers)),layers); ax.set_title(name); ax.set_ylabel('Condition similarity')
    axes[-1].set_xlabel('Decoder block (zero-based); blue=allowed, orange=restricted')
    savefig(fig,args.out,'A_layerwise_scores')
    H=torch.stack([r['last'] for r in saved['activations'][saved['embedding_layer']]]).numpy()
    coordinate_path = args.out/'tsne_coordinates.json'
    if coordinate_path.exists():
        previous = json.loads(coordinate_path.read_text())
        if [r['sample_id'] for r in previous] != saved['sample_ids']:
            raise ValueError('Existing t-SNE coordinate identities differ')
        coordinates = np.array([[r['x'],r['y']] for r in previous])
    else:
        coordinates=TSNE(n_components=2,perplexity=30,init='pca',learning_rate='auto',
                         random_state=42,max_iter=1000).fit_transform(H)
        write(coordinate_path,[{'sample_id':r['sample_id'],'x':float(x),'y':float(y)}
            for r,(x,y) in zip(rows,coordinates)])
    records=read(args.out/'figure4_diagnostics.jsonl')
    by_id={r['sample_id']:r for r in records}
    from scipy.spatial import Delaunay
    from sklearn.neighbors import NearestNeighbors
    # One geometry, bandwidth, support mask and color scale for all modes.
    neighbors = min(20, len(coordinates)-1)
    distances = NearestNeighbors(n_neighbors=neighbors+1).fit(coordinates).kneighbors(coordinates)[0]
    bandwidth = max(float(np.median(distances[:, -1])), 1e-8)
    low, high = coordinates.min(0), coordinates.max(0)
    padding = (high-low)*.05
    gx, gy = np.meshgrid(np.linspace(low[0]-padding[0], high[0]+padding[0], 200),
                         np.linspace(low[1]-padding[1], high[1]+padding[1], 200))
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    supported = Delaunay(coordinates).find_simplex(grid) >= 0
    nearest = NearestNeighbors(n_neighbors=1).fit(coordinates).kneighbors(grid)[0][:,0]
    supported &= nearest <= 2*bandwidth
    surfaces = {}
    fig,axes=plt.subplots(1,3,figsize=(16,5.5),sharex=True,sharey=True)
    colors={'allow':'#2166ac','refuse':'#b2182b',None:'#777777'}
    for ax,mode in zip(axes,MODES):
        actions = [by_id[row['sample_id']][mode+'_action'] for row in rows]
        values = refusal_surface(coordinates, actions, grid, bandwidth)
        surfaces[mode] = values.reshape(gx.shape)
        field = np.ma.masked_where(~supported.reshape(gx.shape), surfaces[mode])
        contour = ax.contourf(gx, gy, field, levels=np.linspace(0,1,13),
                             cmap='RdYlBu_r', vmin=0, vmax=1, alpha=.65)
        for row,xy,action in zip(rows,coordinates,actions):
            ax.scatter(*xy,color=colors[action],marker='o' if row['expected_action']=='allow' else '^',
                       s=22,edgecolors='white',linewidths=.35,zorder=3)
        ax.set_title(mode); ax.set_xlabel('t-SNE 1')
    axes[0].set_ylabel('t-SNE 2')
    fig.subplots_adjust(top=.78, right=.87, wspace=.12)
    cax=fig.add_axes([.89,.17,.013,.54])
    fig.colorbar(contour,cax=cax,ticks=[0,.25,.5,.75,1],label='Smoothed observed refusal frequency')
    fig.suptitle('Shared t-SNE coordinates; background = Gaussian-smoothed judged refusal outcomes\n'
                 'Blue=allow; red=refuse; gray=missing/error. Circle=expected allow; triangle=expected refuse.\n'
                 'White regions lack geometric support. Exploratory surface, not calibrated model probability.', fontsize=10)
    savefig(fig,args.out,'C_shared_tsne')
    np.savez_compressed(args.out/'refusal_surfaces.npz',x=gx,y=gy,supported=supported.reshape(gx.shape),**surfaces)
    write(args.out/'refusal_surface_settings.json', {
        'method':'Gaussian kernel weighted observed refusal fraction', 'bandwidth':bandwidth,
        'bandwidth_rule':f'median distance to {neighbors}th neighbor over all coordinates',
        'grid_shape':[200,200], 'support':'shared convex hull and nearest distance <= 2 * bandwidth',
        'shared_coordinates':True,'shared_bandwidth':True,'color_limits':[0,1],
        'unknown_actions':'excluded, not treated as allow',
        'valid_action_counts':{m:sum(by_id[r['sample_id']][m+'_action'] is not None for r in rows) for m in MODES},
        'coordinate_sha256':sha256(coordinate_path)})
    write(args.out/'embedding_settings.json',{'layer':saved['embedding_layer'],'pool':'last',
        'perplexity':30,'init':'pca','learning_rate':'auto','seed':42,'max_iter':1000,
        'one_embedding_for_all_modes':True})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['offline','extract','plot'],required=True)
    p.add_argument('--data',type=Path,default=HERE/'requests.jsonl')
    p.add_argument('--policies',type=Path,default=HERE/'policies.jsonl')
    p.add_argument('--outputs',type=Path,default=HERE.parent/'results/steering/validation_batched.jsonl')
    p.add_argument('--judgments',type=Path,default=HERE.parent/'results/steering/validation_batched.judgments.jsonl')
    p.add_argument('--cache',type=Path,default=HERE.parent/'results/steering/pilot_calibration.pt')
    p.add_argument('--out',type=Path,default=HERE.parent/'results/steering/experiment4')
    p.add_argument('--layers',type=int,nargs='+',default=[10,14,18,22])
    p.add_argument('--embedding-layer',type=int,default=18)
    p.add_argument('--gpu-memory-fraction',type=float,default=.4)
    args=p.parse_args()
    {'offline':diagnostics,'extract':extract,'plot':plot_activations}[args.phase](args)


if __name__=='__main__':
    main()

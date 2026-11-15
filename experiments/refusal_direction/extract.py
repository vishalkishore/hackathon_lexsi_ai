"""Residual-stream activations of base Aya at the end of the prompt (forward passes only, no generation).

Two positions per layer: the last prompt token, and the mean over the template tokens after the final user turn
(<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>). Layer index = hidden_states index
(0 = embeddings, l = output of decoder layer l; the last one is after the final norm).

  single prompts: saves full activations  acts.pt  [n, layers+1, 2, d] float16
  depth prompts:  saves projections only  proj.pt  [n, layers+1, 2, scripts] onto --directions
Never overwrites; --limit-pairs N keeps N correction + N heldout pairs (smoke test).
"""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import subprocess

import torch

from prompts import build

MODEL = 'CohereLabs/aya-expanse-8b'
REVISION = '5062468bf9bc0c6035fd64e06274333ec127d980'
POSITIONS = ('last', 'tail_mean')


def limit(rows, n):
    keep = set()
    for split in ('correction', 'heldout'):
        keep |= set(sorted({r['pair_id'] for r in rows if r['split'] == split})[:n])
    return [r for r in rows if r['pair_id'] in keep]


def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    tok.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, dtype=torch.bfloat16, device_map='cuda')
    return tok, model.eval()


def encode(tok, rows):
    eot = tok.convert_tokens_to_ids('<|END_OF_TURN_TOKEN|>')
    out = []
    for r in rows:
        text = tok.apply_chat_template(r['messages'], tokenize=False, add_generation_prompt=True)
        ids = tok(text, add_special_tokens=False)['input_ids']
        tail = len(ids) - max(i for i, t in enumerate(ids) if t == eot)
        out.append((ids, tail))
    return out


@torch.no_grad()
def activations(tok, model, encoded, batch_size, reduce=None):
    """Yields per-prompt [layers+1, 2, d] (or reduce(...) of it) in the input order."""
    order = sorted(range(len(encoded)), key=lambda i: -len(encoded[i][0]))
    results = [None] * len(encoded)
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        batch = tok.pad({'input_ids': [encoded[i][0] for i in idx]}, return_tensors='pt').to(model.device)
        hs = torch.stack(model(**batch, output_hidden_states=True).hidden_states, 1)  # [b, L+1, T, d]
        for j, i in enumerate(idx):
            tail = encoded[i][1]
            v = torch.stack([hs[j, :, -1], hs[j, :, -tail:].float().mean(1).to(hs.dtype)], 1)  # [L+1, 2, d]
            results[i] = (reduce(v) if reduce else v).cpu()
        print(f'{min(start + batch_size, len(order))}/{len(order)}', flush=True)
    return torch.stack(results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prompts', choices=['single', 'depth'], required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--directions', type=Path, help='directions.pt from analyze.py fit (required for depth)')
    parser.add_argument('--limit-pairs', type=int)
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    rows = build(args.prompts)
    if args.limit_pairs:
        rows = limit(rows, args.limit_pairs)
    args.out.mkdir(parents=True, exist_ok=False)
    tok, model = load_model()
    encoded = encode(tok, rows)
    print(len(rows), 'prompts; tail lengths', sorted({t for _, t in encoded}),
          '; tail tokens', tok.convert_ids_to_tokens(encoded[0][0][-encoded[0][1]:]), flush=True)
    if args.prompts == 'single':
        torch.save(activations(tok, model, encoded, args.batch_size).half(), args.out / 'acts.pt')
    else:
        dirs = torch.load(args.directions)
        stack = torch.stack([dirs['r_hat'][s] for s in dirs['scripts']], -1).cuda()  # [L+1, 2, d, S]
        proj = activations(tok, model, encoded, args.batch_size, lambda v: torch.einsum('lpd,lpds->lps', v.float(), stack))
        torch.save(proj, args.out / 'proj.pt')
    with (args.out / 'prompts.jsonl').open('x', encoding='utf-8') as f:
        f.writelines(json.dumps({k: v for k, v in r.items() if k != 'messages'}, ensure_ascii=False) + '\n' for r in rows)
    meta = {'utc': datetime.now(timezone.utc).isoformat(), 'model': MODEL, 'revision': REVISION, 'prompts': args.prompts,
            'n': len(rows), 'limit_pairs': args.limit_pairs, 'positions': POSITIONS, 'directions': str(args.directions),
            'layers': model.config.num_hidden_layers + 1, 'dtype': 'bfloat16 forward',
            'git': subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip() or None,
            'versions': {k: version(k) for k in ('torch', 'transformers')}, 'gpu': torch.cuda.get_device_name()}
    (args.out / 'meta.json').write_text(json.dumps(meta, indent=2))
    print('saved', args.out)


if __name__ == '__main__':
    main()

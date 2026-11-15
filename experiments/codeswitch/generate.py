"""Generate Aya replies to the final (Hindi-switch) turn with vLLM, temperature 0 (greedy); never overwrites.

One model load serves both modes: 'multi' (English opener then switch) and 'single' (request only).
"""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path

SYSTEM = 'You are a helpful assistant.'
REVISION = '5062468bf9bc0c6035fd64e06274333ec127d980'


def messages(row, single_turn=False):
    turns = row.get('opener_turns') or [{'user': row['opener_user'], 'assistant': row['opener_assistant']}]
    opener = [] if single_turn else [m for t in turns for m in ({'role': 'user', 'content': t['user']},
                                                                {'role': 'assistant', 'content': t['assistant']})]
    return [{'role': 'system', 'content': SYSTEM}, *opener, {'role': 'user', 'content': row['request']}]


def read_partial(path):
    """IDs already saved in a partial output; a truncated last line from an interrupted write is dropped."""
    if not path.exists():
        return set()
    good = []
    for line in path.read_text(encoding='utf-8').splitlines():
        try:
            good.append(json.loads(line))
        except json.JSONDecodeError:
            break
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in good), encoding='utf-8')
    return {r['conversation_id'] for r in good}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--model', default='CohereLabs/aya-expanse-8b')
    parser.add_argument('--condition', default='base')
    parser.add_argument('--modes', nargs='+', choices=['multi', 'single'], default=['multi', 'single'])
    parser.add_argument('--max-new-tokens', type=int, default=8192)
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.85)
    parser.add_argument('--chunk-size', type=int, default=2000, help='conversations per save; reruns resume from the .partial file')
    args = parser.parse_args()
    rows = [json.loads(l) for l in (args.run_dir / 'conversations.jsonl').read_text().splitlines()]
    outputs = {m: args.run_dir / f'{args.condition}{"-single" if m == "single" else ""}-full.jsonl' for m in args.modes}
    for path in outputs.values():
        if path.exists():
            raise FileExistsError(path)
    from vllm import LLM
    kw = {'revision': REVISION, 'tokenizer_revision': REVISION} if args.model == 'CohereLabs/aya-expanse-8b' else {}
    llm = LLM(model=args.model, dtype='bfloat16', seed=42, max_model_len=8192,
              gpu_memory_utilization=args.gpu_memory_utilization, **kw)
    params = llm.get_default_sampling_params()  # model generation_config, else vLLM defaults
    params.max_tokens, params.seed = args.max_new_tokens, 42
    params.temperature, params.top_p = 0.0, 1.0  # deterministic (greedy) decoding
    tok = llm.get_tokenizer()
    for mode, path in outputs.items():
        single = mode == 'single'
        settings = {'backend': 'vllm', 'temperature': params.temperature, 'top_p': params.top_p, 'top_k': params.top_k, 'max_new_tokens': args.max_new_tokens, 'seed': 42,
                    'system_prompt': SYSTEM, 'single_turn': single, 'max_model_len': 8192,
                    'chat_template': tok.chat_template}
        partial = path.with_name(path.name + '.partial')  # renamed to the final name only when complete
        done = read_partial(partial)
        todo = [r for r in rows if r['conversation_id'] not in done]
        print(f'{mode}: {len(done)} already saved, {len(todo)} to generate', flush=True)
        with partial.open('a', encoding='utf-8') as f:
            for start in range(0, len(todo), args.chunk_size):
                chunk = todo[start:start + args.chunk_size]
                results = llm.chat([messages(r, single) for r in chunk], params)
                for r, res in zip(chunk, results):
                    o = res.outputs[0]
                    f.write(json.dumps({**r, 'single_turn': single, 'model_condition': args.condition, 'model': args.model,
                                        'model_revision': kw.get('revision'), 'response': o.text,
                                        'finish_reason': o.finish_reason, 'generated_tokens': len(o.token_ids),
                                        'generation': settings, 'utc': datetime.now(timezone.utc).isoformat(),
                                        'versions': {k: version(k) for k in ('torch', 'transformers', 'vllm')}},
                                       ensure_ascii=False) + '\n')
                f.flush()
                print(f'{mode}: {len(done) + start + len(chunk)}/{len(rows)} saved', flush=True)
        assert len(read_partial(partial)) == len(rows), 'partial file incomplete'
        partial.rename(path)
        print('Saved', path, flush=True)


if __name__ == '__main__':
    main()

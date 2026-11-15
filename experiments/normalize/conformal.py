"""Split-conformal gate threshold for automatic-Devanagari input (Task 1.4d), from recognize.jsonl (no model calls).

Calibration scores: layer-19 last-token projections on r_hat_en of the BENIGN prompts of the 250 correction pairs that
are not in the 83-pair sweep, with the automatic-Devanagari final message (same v2 depth-8 prefix).
tau = the ceil((n + 1)(1 - alpha))-th smallest score. For a new benign prompt exchangeable with the calibration set,
P(score > tau) <= alpha (marginal guarantee, benign false-fire rate). alpha = 0.05.
"""
import argparse
import json
import math
from pathlib import Path


def conformal_tau(scores, alpha):
    s = sorted(scores)
    k = math.ceil((len(s) + 1) * (1 - alpha))
    if k > len(s):
        raise ValueError(f'n = {len(s)} too small for alpha = {alpha}')
    return s[k - 1], k


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recognize', type=Path, required=True, help='runs/normalize_v1/recognize.jsonl')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--alpha', type=float, default=0.05)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    rows = [json.loads(line) for line in args.recognize.read_text().splitlines()]
    cal = [r['auto'] for r in rows if r['request_type'] == 'benign' and not r['in_sweep']]
    tau, k = conformal_tau(cal, args.alpha)
    out = {'layer': 19, 'position': 'last', 'input': 'automatic Devanagari (IndicXlit)', 'alpha': args.alpha,
           'n_calibration': len(cal), 'rank': k, 'tau': {'hi-Latn': tau},
           'calibration': 'benign, 250 correction pairs not in the sweep, v2 depth 8'}
    args.out.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()

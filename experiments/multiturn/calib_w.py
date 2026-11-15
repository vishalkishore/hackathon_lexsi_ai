"""EXPLORATORY Step D method (user Decision 1, 2026-11-11): w (b, lambda 0, unconstrained, M1 code) fit on the 250
non-sweep correction pairs; split-conformal threshold per (script, depth) on the 83 sweep pairs' BENIGN scores,
k = ceil(84 * 0.95) = 80. Checks before Step E: calibration benign fire <= 4/83 per cell; hi-Latn d8 sweep catch
within 0.05 of Step B' (base 0.937, Read 0.969). Exact split conformal for base; approximate for Read (the Read edit
was fit on all 333 pairs, the sweep pairs included). Writes directions.pt, thresholds.json, calib_w.json.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "refusal_direction"))

import m1  # noqa: E402
from cache import DEPTHS, LAYERS, SCRIPTS, environment  # noqa: E402
from gen import conformal  # noqa: E402
from ov_common import SCORE_LAYER, catch_rate, sweep_pairs  # noqa: E402

B2_CATCH = {"base": 0.937, "read": 0.969}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--threads", type=int, default=16)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "env.json").write_text(json.dumps(environment(args.out) | {"cache": str(args.cache)}, indent=1))
    pairs = json.loads((args.cache / "pairs.json").read_text())
    sw = sweep_pairs()
    fit_idx = np.array([i for i, q in enumerate(pairs) if q not in sw])
    cal_idx = np.array([i for i, q in enumerate(pairs) if q in sw])
    assert len(fit_idx) == 250 and len(cal_idx) == 83
    res, dirs, thr = {}, {}, {}
    for name in ("base", "read"):
        X = torch.load(args.cache / name / "h.pt")[..., LAYERS.index(SCORE_LAYER), :].double()
        w, _, _, resid = m1.solve(m1.matrices(X, fit_idx), "b", 0.0)
        if torch.einsum("psdf,f->", X[fit_idx, 1] - X[fit_idx, 0], w) < 0:
            w = -w
        sc = torch.einsum("pasdf,f->pasd", X[cal_idx], w)
        cells = {}
        for s, script in enumerate(SCRIPTS):
            for d, depth in enumerate(DEPTHS):
                c = conformal(sc[:, 0, s, d].tolist(), 0.05)
                c["benign_fire"] = int((sc[:, 0, s, d] > c["c"]).sum())
                c["harmful_fire"] = int((sc[:, 1, s, d] > c["c"]).sum())
                c["catch"] = catch_rate(sc[:, 1, s, d].tolist(), sc[:, 0, s, d].tolist())
                assert c["k"] == 80
                cells[f"{script}|{depth}"] = c
        catch = cells["hi-Latn|8"]["catch"]
        res[name] = {"residual": resid, "catch_hiLatn_d8": catch, "B2_catch": B2_CATCH[name],
                     "checks": {"benign fire <= 4/83 every cell": all(v["benign_fire"] <= 4 for v in cells.values()),
                                "catch within 0.05 of B'": abs(catch - B2_CATCH[name]) <= 0.05, "residual < 1e-10": resid < 1e-10},
                     "guarantee": "exact split conformal" if name == "base" else
                     "approximate: the Read edit was fit on all 333 pairs, the calibration (sweep) pairs included"}
        res[name]["pass"] = all(res[name]["checks"].values())
        dirs[name], thr[name] = w, cells
        print(name, json.dumps({k: res[name][k] for k in ("catch_hiLatn_d8", "checks", "pass")}), flush=True)
    torch.save({"w_base": dirs["base"], "w_read": dirs["read"], "version": "b", "lambda": 0.0, "layer": SCORE_LAYER,
                "fit": "250 non-sweep correction pairs"}, args.out / "directions.pt")
    (args.out / "thresholds.json").write_text(json.dumps({"alpha": 0.05, "calibration": "83 sweep pairs, benign", **thr}, indent=1))
    (args.out / "calib_w.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

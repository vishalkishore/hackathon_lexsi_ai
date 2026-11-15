"""Controls Step 2 (amended, docs/prereg_controls.md): random-detector thresholds on CPU, with calib_w's recipe.
For each model (base, read), seed s in --seeds, depth in 0/8: c_v = conformal(v_s . h) over the 83 sweep pairs' benign
hi-Latn scores at layer 19, from runs/multiturn_v2/m0_cache/<model>/h.pt (the cache that set c_w). Check: c_w recomputed
from the same cache with calib_w w equals the stored c_w within 1e-6, else stop. Writes runs/controls/controls_cv.json."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "multiturn"), str(HERE.parent / "refusal_direction")]
from cache import DEPTHS, LAYERS, SCRIPTS  # noqa: E402
from gen import conformal  # noqa: E402
from ov_common import SCORE_LAYER, sweep_pairs  # noqa: E402

CACHE, CALIB, OUT = Path("runs/multiturn_v2/m0_cache"), Path("runs/multiturn_v2/calib_w"), Path("runs/controls/controls_cv.json")


def unit(seed, n=4096):
    v = torch.randn(n, generator=torch.Generator().manual_seed(seed))
    return v / v.norm()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    args = ap.parse_args()
    pairs = json.loads((CACHE / "pairs.json").read_text())
    sw = sweep_pairs()
    cal = np.array([i for i, q in enumerate(pairs) if q in sw])
    assert len(cal) == 83
    thr, dirs = json.loads((CALIB / "thresholds.json").read_text()), torch.load(CALIB / "directions.pt")
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    s_i = SCRIPTS.index("hi-Latn")
    for name in ("base", "read"):
        X = torch.load(CACHE / name / "h.pt")[..., LAYERS.index(SCORE_LAYER), :].double()[cal, 0, s_i]  # [83, depths, F] benign
        for dep in (0, 8):
            H = X[:, DEPTHS.index(dep)]
            cw = conformal((H @ dirs[f"w_{name}"].double()).tolist(), 0.05)["c"]
            stored = thr[name][f"hi-Latn|{dep}"]["c"]
            assert abs(cw - stored) < 1e-6, (name, dep, cw, stored)
            out[f"check|{name}|{dep}"] = {"recomputed": cw, "stored": stored}
            for s in args.seeds:
                out[f"RD{s}|{name}|{dep}"] = conformal((H @ unit(s).double()).tolist(), 0.05)
        print(name, "done", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v.get("c", v) for k, v in out.items()}, indent=1))


if __name__ == "__main__":
    main()

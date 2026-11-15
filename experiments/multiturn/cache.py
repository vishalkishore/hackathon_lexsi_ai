"""Stage M0 cache: residual activations at the last prompt token, base and Read model. Forward passes only.

Design: 333 correction pairs x harm {benign, harmful} x script {en, hi-Deva, hi-Latn, en-hi} x depth {0, 2, 5, 8}
(depth 0 = single-turn form, depth > 0 = v2 pool), 10,656 prompts per model.

Layer index = hidden_states index (0 = embedding output, l = output of layers[l-1]). This is the convention of
r_hat (layer 19 = output of layers[18]). The Read edit changes layers[16].mlp.down_proj, so it first shows at index 17.

Outputs in --out/<model>/ (model = base, read):
  h.pt     float32 [P, A, S, D, len(LAYERS), 4096]; bf16 forward, values cast to fp32 at storage.
  ksum.pt  float64 [len(KEY_BLOCKS), A, S, D, 14336]: sum over pairs of the down_proj input (MLP key) of
           layers[b] at the last token. M0.6 is linear in the key, so the pair sum is enough.
  pairs.json  pair_id order of axis P.
Checks (written to checks.json, the run stops on failure):
  stored_scores: layer-19 r_hat score equals the stored scores within 0.15. Base: step2/scores.pt (all stored cells).
    Read: read_final/cal_scores.json 'in_sample' (final edit, benign non-sweep hi-Latn and en-hi). The step-2 Read
    scores are out-of-fold edits, so they are not a reference for delta_final. Batches are formed per (script, depth)
    cell, longest first, as in step 2: batch membership changes bf16 values. Enforced only on the full design.
  mlp_sum (M0.6a): sum_j k_j W_down[:, j] equals the down_proj output, relative error < 1e-2.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "refusal_direction"))

from extract import encode, load_model  # noqa: E402
from ov_common import SCORE_LAYER, r_hat, rows_for, sweep_pairs  # noqa: E402
from prompts import ROOT  # noqa: E402
from read_pilot import WeightEdit  # noqa: E402

HARMS = ("benign", "harmful")
SCRIPTS = ("en", "hi-Deva", "hi-Latn", "en-hi")
DEPTHS = (0, 2, 5, 8)
LAYERS = sorted(set(range(0, 31, 2)) | set(range(12, 20)))
KEY_BLOCKS = tuple(range(11, 19))  # layers[b] writes into hidden index b + 1 = 12..19
DELTA = ROOT / "runs/overnight_v2/read_final/delta_final.pt"
STEP2 = ROOT / "runs/overnight_v2/step2"
READ_SCORES = ROOT / "runs/overnight_v2/read_final/cal_scores.json"
TOL_BF16 = 0.15


class Stop(Exception):
    pass


def design_rows(limit_pairs=None):
    """All rows with their (p, a, s, d) index. Every cell must be present exactly once."""
    pairs = sorted({r["pair_id"] for r in rows_for("en", 0)})
    if limit_pairs:
        pairs = pairs[:limit_pairs]
    p_index = {p: i for i, p in enumerate(pairs)}

    rows = []
    for s, script in enumerate(SCRIPTS):
        for d, depth in enumerate(DEPTHS):
            for r in rows_for(script, depth):
                if r["pair_id"] in p_index:
                    r["cell"] = (p_index[r["pair_id"]], HARMS.index(r["request_type"]), s, d)
                    rows.append(r)

    cells = {r["cell"] for r in rows}
    assert len(cells) == len(rows) == len(pairs) * 2 * 4 * 4, "design is not balanced"
    return pairs, rows


@torch.no_grad()
def collect(tok, model, rows, batch_size):
    store = {}
    hooks = [model.model.embed_tokens.register_forward_hook(lambda _m, _i, o: store.__setitem__(0, o[:, -1]))]
    for layer in LAYERS[1:]:
        def keep(_m, _i, out, layer=layer):
            store[layer] = (out[0] if isinstance(out, tuple) else out)[:, -1]
            if layer == LAYERS[-1]:
                raise Stop
        hooks.append(model.model.layers[layer - 1].register_forward_hook(keep))
    for b in KEY_BLOCKS:
        def keep_key(_m, inputs, out, b=b):
            store[("k", b)] = inputs[0][:, -1]
            store[("mlp", b)] = out[:, -1]
        hooks.append(model.model.layers[b].mlp.down_proj.register_forward_hook(keep_key))

    P = 1 + max(r["cell"][0] for r in rows)
    h = torch.zeros(P, 2, 4, 4, len(LAYERS), model.config.hidden_size)
    ksum = torch.zeros(len(KEY_BLOCKS), 2, 4, 4, model.config.intermediate_size, dtype=torch.float64)
    mlp_err = 0.0

    encoded = encode(tok, rows)
    batches = []
    for cell in sorted({r["cell"][2:] for r in rows}):
        in_cell = [i for i, r in enumerate(rows) if r["cell"][2:] == cell]
        order = sorted(in_cell, key=lambda i: -len(encoded[i][0]))
        batches += [order[k:k + batch_size] for k in range(0, len(order), batch_size)]
    try:
        for n_done, idx in enumerate(batches):
            batch = tok.pad({"input_ids": [encoded[i][0] for i in idx]}, return_tensors="pt").to(model.device)
            try:
                model.model(**batch)
            except Stop:
                pass

            for j, i in enumerate(idx):
                p, a, s, d = rows[i]["cell"]
                h[p, a, s, d] = torch.stack([store[layer][j].float() for layer in LAYERS]).cpu()
                for bi, b in enumerate(KEY_BLOCKS):
                    ksum[bi, a, s, d] += store[("k", b)][j].double().cpu()

            # M0.6a: channel writes sum to the MLP output (bf16 forward vs fp32 sum).
            if n_done == 0:
                for b in KEY_BLOCKS:
                    w = model.model.layers[b].mlp.down_proj.weight.float()
                    summed = store[("k", b)].float() @ w.T
                    ref = store[("mlp", b)].float()
                    mlp_err = max(mlp_err, float((summed - ref).norm() / ref.norm()))
            print(f"batch {n_done + 1}/{len(batches)}", flush=True)
    finally:
        for hook in hooks:
            hook.remove()
    return h, ksum, mlp_err


def stored_reference(name):
    """(script, depth, pair_id, harm) -> stored layer-19 score."""
    ref = {}
    if name == "base":
        stored = torch.load(STEP2 / "scores.pt", weights_only=False)["base"]
        prompts = [json.loads(line) for line in (STEP2 / "prompts.jsonl").read_text().splitlines()]
        for key, values in stored.items():
            script, depth = key.split("|")
            cell_rows = [x for x in prompts if x["script"] == script and x["depth"] == int(depth)]
            assert len(cell_rows) == len(values)
            for x, v in zip(cell_rows, values):
                ref[(script, int(depth), x["pair_id"], x["request_type"])] = v
    else:
        stored = json.loads((READ_SCORES).read_text())
        sweep = sweep_pairs()
        for key, values in stored.items():
            script, depth = key.split("|")
            if script not in SCRIPTS:
                continue
            cell_rows = [x for x in rows_for(script, int(depth))
                         if x["request_type"] == "benign" and x["pair_id"] not in sweep]
            assert len(cell_rows) == len(values["in_sample"])
            for x, v in zip(cell_rows, values["in_sample"]):
                ref[(script, int(depth), x["pair_id"], "benign")] = v
    return ref


def score_check(h, pairs, name):
    """Max |cached layer-19 score - stored score| over the stored prompts present in the cache."""
    r = r_hat("cpu").double()
    p_index = {p: i for i, p in enumerate(pairs)}
    li = LAYERS.index(SCORE_LAYER)

    worst, n = 0.0, 0
    for (script, depth, pair_id, harm), v in stored_reference(name).items():
        if pair_id not in p_index:
            continue
        cell = (p_index[pair_id], HARMS.index(harm), SCRIPTS.index(script), DEPTHS.index(depth))
        worst = max(worst, abs(float(h[cell][li].double() @ r) - float(v)))
        n += 1
    return {"max_abs_diff": worst, "n": n, "tol": TOL_BF16, "pass": n > 0 and worst < TOL_BF16}


def environment(out):
    code = sorted((HERE.parent / "refusal_direction").glob("*.py")) + sorted(HERE.glob("*.py"))
    import transformers
    return {
        "utc": datetime.now(timezone.utc).isoformat(),
        "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip(),
        "git_dirty": bool(subprocess.run(["git", "status", "--porcelain", "circuit"], capture_output=True, text=True,
                                         cwd=HERE).stdout.strip()),
        "code_sha256": {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in code},
        "nvidia_smi_L": subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.strip(),
        "cuda_visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__, "transformers": transformers.__version__, "cuda": torch.version.cuda,
        "python": sys.version.split()[0],
        "argv": sys.argv, "out": str(out),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit-pairs", type=int, help="smoke test: first N pairs only")
    args = parser.parse_args()

    from setproctitle import setproctitle
    setproctitle("aya_alignment:M0_cache")
    pairs, rows = design_rows(args.limit_pairs)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "env-cache.json").write_text(json.dumps(environment(args.out), indent=1))
    (args.out / "pairs.json").write_text(json.dumps(pairs))
    print(len(pairs), "pairs,", len(rows), "prompts per model; layers", LAYERS, flush=True)

    tok, model = load_model()
    edit = WeightEdit(model, DELTA)
    checks = {"delta": str(DELTA), "layers": LAYERS, "key_blocks": KEY_BLOCKS, "realized_edit": edit.realized()}

    for name in ("base", "read"):
        if name == "read":
            edit.apply(1.0)
        h, ksum, mlp_err = collect(tok, model, rows, args.batch_size)
        folder = args.out / name
        folder.mkdir()
        torch.save(h, folder / "h.pt")
        torch.save(ksum, folder / "ksum.pt")
        checks[name] = {
            "stored_scores": score_check(h, pairs, name),
            "mlp_sum_M0.6a": {"max_rel_err": mlp_err, "tol": 1e-2, "pass": mlp_err < 1e-2},
        }
        print(name, json.dumps(checks[name]), flush=True)
    edit.revert()

    (args.out / "checks.json").write_text(json.dumps(checks, indent=1))
    failed = [f"{n}:{k}" for n in ("base", "read") for k, v in checks[n].items() if not v["pass"]]
    if args.limit_pairs:
        # Smoke batches cannot match the stored full-cell batches, so the score check is reported only.
        failed = [f for f in failed if not f.endswith("stored_scores")]
    if failed:
        sys.exit(f"VERIFY failed: {failed}")
    print("cache done", args.out)


if __name__ == "__main__":
    main()

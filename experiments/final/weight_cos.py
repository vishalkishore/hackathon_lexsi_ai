"""Controls Step 4 (docs/prereg_controls.md): cosine between the romanized and the Devanagari state of the same held-out
conversation (v2 held-out, 83 pairs, harmful + benign, depths 0 / 8; Devanagari = same conversation, final message =
the v2 hi-Deva text), at the output of layer 16 (hidden_states[17]) and layer 19 (hidden_states[19]), last prompt token
and final-user-turn mean, before and after the Read weight edit (runs/overnight_v2/read_final/delta_final.pt).
Forward passes only. Bootstrap over pairs (4000, seed 0). Writes runs/controls/step4_cos.csv + .md."""
import copy
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]
from final_gen import READ, WeightEdit, load_model, rows_of  # noqa: E402
from extract import encode  # noqa: E402
from intervene import final_turn_start  # noqa: E402

OUT = Path("runs/controls")
LAYERS = (17, 19)  # hidden_states index: 17 = output of layer 16 (decoder module 16), 19 = output of layers[18]


def pairs_of(d):
    lat = rows_of("v2", "hi-Latn", d)
    deva = {r["item_id"]: r["request"] for r in rows_of("v2", "hi-Deva", d)}
    gold = []
    for r in copy.deepcopy(lat):
        assert r["messages"][-1] == {"role": "user", "content": r["request"]}
        r["messages"][-1] = {"role": "user", "content": deva[r["item_id"]]}
        gold.append(r)
    return lat, gold


@torch.no_grad()
def states(tok, model, rows):
    out = {}
    for r, (ids, _) in zip(rows, encode(tok, rows)):
        hs = model(torch.tensor([ids], device=model.device), output_hidden_states=True).hidden_states
        a = final_turn_start(tok, ids)
        out[r["item_id"]] = {L: (hs[L][0, -1].float().cpu(), hs[L][0, a:].float().mean(0).cpu()) for L in LAYERS}
    return out


def main():
    tok, model = load_model()
    edit = WeightEdit(model, READ / "delta_final.pt")
    recs = []
    for d in (0, 8):
        lat, gold = pairs_of(d)
        S = {}
        for name in ("before", "after"):
            edit.apply(1.0) if name == "after" else edit.revert()
            S[name] = (states(tok, model, lat), states(tok, model, gold))
        edit.revert()
        pid = {r["item_id"]: r["pair_id"] for r in lat}
        for L in LAYERS:
            for pi, pos in enumerate(("last token", "final-turn mean")):
                cos = {n: {i: float(torch.nn.functional.cosine_similarity(S[n][0][i][L][pi], S[n][1][i][L][pi], dim=0)) for i in pid} for n in S}
                groups = sorted(set(pid.values()))
                by = {n: np.array([np.mean([cos[n][i] for i in pid if pid[i] == g]) for g in groups]) for n in cos}
                idx = np.random.default_rng(0).integers(0, len(groups), (4000, len(groups)))
                rec = {"depth": d, "layer_output": L - 1 if L == 17 else 19, "hidden_states_index": L, "position": pos, "n_pairs": len(groups)}
                for n in ("before", "after"):
                    rec[f"cos_{n}"], (rec[f"cos_{n}_ci_low"], rec[f"cos_{n}_ci_high"]) = by[n].mean(), np.quantile(by[n][idx].mean(1), [0.025, 0.975])
                diff = by["after"] - by["before"]
                rec["diff"], (rec["diff_ci_low"], rec["diff_ci_high"]) = diff.mean(), np.quantile(diff[idx].mean(1), [0.025, 0.975])
                recs.append(rec)
                print(rec, flush=True)
    df = pd.DataFrame(recs)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "step4_cos.csv", index=False)
    (OUT / "step4_cos.md").write_text("# Step 4: cos(romanized, Devanagari state), held-out, before / after the weight edit\n\n"
                                      + df.round(4).to_markdown(index=False) + "\n")


if __name__ == "__main__":
    main()

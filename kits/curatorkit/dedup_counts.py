"""CuratorKIT 0.1.22 filters on both sets, counts next to ours (no GPU needed). Inputs loaded as ayasafe.data.pairs does
(pinned revisions ayasafe.data.REVS; ayasafe.data.clean). Dev = the 416-pair set (heretic-org/Multilingual-Harmless-
Harmful, english config, harmful + harmless). OOD = the fresh harmful candidates (HarmBench standard + Aya red-teaming
Hindi, English text, len > 10). Filters (curatorkit.normalizers): ExactDeduplicator; MinHashDeduplicator (Jaccard 0.85);
EmbeddingDeduplicator (embeddinggemma-300m at ayasafe.data.EMB's revision): within-set at 0.90, and the leak check OOD
vs dev at 0.85 (index = all dev english + hindi texts; counted with the deduplicator's own _check_against_index).
Ours encodes with prompt_name="STS"; CuratorKIT encodes the raw text, hence 8 vs 16 leaks and 61 vs 79 near-dups.
  PYTHONPATH=submission/repo HF_HUB_OFFLINE=1 python submission/repo/kits/curatorkit/dedup_counts.py"""
import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from curatorkit.normalizers.dedup import ExactDeduplicator, MinHashDeduplicator
from curatorkit.normalizers.embedding_dedup import EmbeddingDeduplicator
from curatorkit.schema import DataSample
from datasets import load_dataset
from huggingface_hub import snapshot_download

from ayasafe.data import EMB, REVS, clean

HERETIC = "heretic-org/Multilingual-Harmless-Harmful"


def dev_texts(configs):
    return clean([t for c in configs for s in ("harmful", "harmless") for t in load_dataset(HERETIC, c, split=s, revision=REVS[HERETIC])["text"]])


def ood_frame():
    hb = load_dataset("walledai/HarmBench", "standard", split="train", revision=REVS["walledai/HarmBench"])
    aya = load_dataset("CohereLabs/aya_redteaming", split="hindi", revision=REVS["CohereLabs/aya_redteaming"])
    harm = pd.concat([pd.DataFrame({"source_text": hb["prompt"], "english": hb["prompt"]}),
                      pd.DataFrame({"source_text": aya["prompt"],
                                    "english": [s or lit for s, lit in zip(aya["semantic_translation"], aya["literal_translation"])]})],
                     ignore_index=True)
    harm["source_text"], harm["english"] = clean(harm.source_text), clean(harm.english)
    return harm[harm.english.str.len() > 10].reset_index(drop=True)


def samples(texts, tag):
    return [DataSample(task_type="language_modeling", output=t, id=f"{tag}{i}", source_uri=tag) for i, t in enumerate(texts)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("runs/kits/curatorkit_counts.json"))
    args = ap.parse_args()
    emb = snapshot_download(EMB[0], revision=EMB[1], local_files_only=True)  # local path at the pinned revision

    def dedup(texts, tag, thr, index):
        d = EmbeddingDeduplicator(index_dir=index, model=emb, threshold=thr, text_field="output")
        return d.run(samples(texts, tag))
    dev_en, dev_all, harm = dev_texts(["english"]), dev_texts(["english", "hindi"]), ood_frame()
    rows = []
    exact = ExactDeduplicator().run(samples(harm.source_text.tolist(), "ood"))
    ids = {s.id for s in exact}
    ood = harm[[f"ood{i}" in ids for i in range(len(harm))]].reset_index(drop=True)
    rows.append({"set": "OOD harmful", "filter": "exact dedup (source text)", "before": len(harm), "after_curatorkit": len(exact), "after_ours": 1088})
    with tempfile.TemporaryDirectory() as t:
        dedup(dev_all, "dev", 1.01, Path(t))  # index every dev text (threshold > 1: nothing dropped)
        d = EmbeddingDeduplicator(index_dir=Path(t), model=emb, threshold=0.85, text_field="output")
        d._load_index()
        E = d._load_model().encode(ood.english.tolist(), batch_size=d.batch_size, convert_to_numpy=True).astype(np.float32)
        E /= np.where(np.linalg.norm(E, axis=1, keepdims=True) == 0, 1.0, np.linalg.norm(E, axis=1, keepdims=True))
        leaks = sum(bool(d._check_against_index(E[i:i + 1])[0]) for i in range(len(E)))  # run()'s cross-run check
    rows.append({"set": "OOD vs dev", "filter": "leak check, cosine >= 0.85", "before": len(ood), "after_curatorkit": len(ood) - leaks, "after_ours": 1072})
    with tempfile.TemporaryDirectory() as t:
        rows.append({"set": "OOD harmful", "filter": "near-dup within set, cosine >= 0.90", "before": len(ood),
                     "after_curatorkit": len(dedup(ood.english.tolist(), "ood", 0.90, Path(t))), "after_ours": "993 (after the leak step)"})
    rows.append({"set": "OOD harmful", "filter": "MinHash, Jaccard >= 0.85", "before": len(ood),
                 "after_curatorkit": len(MinHashDeduplicator(threshold=0.85).run(samples(ood.english.tolist(), "ood"))), "after_ours": None})
    for name, f in (("exact dedup", ExactDeduplicator()), ("MinHash, Jaccard >= 0.85", MinHashDeduplicator(threshold=0.85))):
        rows.append({"set": "dev (english)", "filter": name, "before": len(dev_en), "after_curatorkit": len(f.run(samples(dev_en, "dev"))), "after_ours": None})
    with tempfile.TemporaryDirectory() as t:
        rows.append({"set": "dev (english)", "filter": "near-dup within set, cosine >= 0.90", "before": len(dev_en),
                     "after_curatorkit": len(dedup(dev_en, "dev", 0.90, Path(t))), "after_ours": None})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=1))
    print(pd.DataFrame(rows).to_string())


if __name__ == "__main__":
    main()

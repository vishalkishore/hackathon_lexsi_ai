# CuratorKIT (curatorkit 0.1.22)

`dedup_counts.py`: `ExactDeduplicator`, `MinHashDeduplicator` (Jaccard 0.85), `EmbeddingDeduplicator` (embeddinggemma-300m;
within set 0.90; leak check of the out-of-distribution set vs the 416-pair set at 0.85). Counts next to ours: exact 1,090 →
1,088 (ours 1,088); leaks 8 (ours 16); near-dups 61 (ours 79). Our pipeline encoded with the STS prompt; without it our
code gives 8 and 61. Script, romanized-script and length checks are our own code (ayasafe.data), not CuratorKIT.
Run: `PYTHONPATH=submission/repo HF_HUB_OFFLINE=1 python submission/repo/kits/curatorkit/dedup_counts.py`.

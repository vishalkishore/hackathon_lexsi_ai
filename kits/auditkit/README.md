# AuditKIT (auditkit 1.1.3) — Phase 1 capability evals and the final-model capability check

| file | features used | produced | stored output (results dataset `tables/kits/`) |
|---|---|---|---|
| `eval_cap.sh` | `auditkit.run_lmeval` (lm-eval engine), `auditkit.RunConfig`, `result.save`; tasks gsm8k (5-shot, temperature 0, 256 tokens), mmlu (5), arc_challenge (25), hellaswag (10), truthfulqa_mc2 (0); bf16, chat template | Phase 1 base capability; final-model GSM8K + MMLU (base vs N-Rw merged) | `phase1_base_auditkit_*.json`; `final/` |

Run: `TASKS="gsm8k mmlu" bash eval_cap.sh <model id or local path> <name> <gpu>` (writes results/<name>/capability).

## Stored-reply scoring (final model)
`gsm8k_stored.py`: `auditkit.evaluate` (`CallableModel` returning the stored replies, `auditkit.scorer` = last number == gold)
+ `auditkit.compare(...).significance()`. Reproduces GSM8K Base 80.06, main method 80.67 (Δ +0.61, p 0.58) exactly.
Run from the research root: `PYTHONPATH=submission/repo python submission/repo/kits/auditkit/gsm8k_stored.py`.

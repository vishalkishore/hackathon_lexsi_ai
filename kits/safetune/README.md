# SafeTune (safetune 0.1.10) — earlier / background work and final-model XSTest

| file | features used | produced | stored output (results dataset `tables/kits/`) |
|---|---|---|---|
| `scripts/eval_safety.sh` | `safetune eval` CLI, vLLM backend; suites advbench (520), harmbench (400), wildjailbreak (500); WildGuard judge (`asr` = non-refusal) | Phase 1 base safety numbers | `phase1_base_safetune_{advbench,harmbench,wildjailbreak}.log` |
| `scripts/eval_xstest_split.py` | `safetune.data.loaders`, `safetune.evaluate` (judge `wildguard`), `safetune.runner.utils.model_utils.load_model` | XSTest safe (250) / unsafe (200) refusal; also the final-model run | `phase1_base_safetune_xstest_split.json`; `final/` |
| `scripts/eval_harmbench.py`, `scripts/eval_cast_safety.py`, `massive_aya/` | `safetune.evaluate`; `safetune.steer.CASTCondition` gates on a fine-tuned model | CAST gate transfer to an SFT model (Phase 1) | none copied (not reported) |
| `steering/cast_experiment.py`, `policy_cast.py`, `experiment4.py` | `safetune.steer`: `CAAModel`, `CAAConfig`, `CASTModel`, `CASTCondition`, `fit_cast_condition`; `_cast_similarity`, `_get_decoder_layers` | policy-steering study (Phase 2): CAST gate diagnostics, OR-gated multi-policy CAST | `phase2_safetune_cast_gate_diagnostic.json` |
| `steering/legal_four_arm.py` | reuse of cached SafeTune CAA / CAST outputs | legal-policy four-arm table | `phase2_safetune_legal_four_arm_{scores.json,summary.md}` |
| `steering/compare_safetune_ibm.py`, `original_cast_*.py` | SafeTune CAST vs IBM activation-steering CAST | coding-tutor CAST comparison (negative) | `phase2_safetune_vs_ibm_cast_summary.md`, `phase2_original_cast_dev_metrics.json` |

Code copied as-is from the research tree (it imports sibling modules from its own folder; run from that folder).
Reported claim without a stored file: "script predicted the expected label in the benchmark" (policy-steering study).

## Final-model kit uses
- `xstest_second_judge.py`: SafeTune's XSTest over-refusal metric (`OverRefusalMetric`, keyword judge) on the stored
  XSTest replies, next to the v4 labels: safe over-refusal 4.8 → 11.6 (Δ +6.8 [+3.2, +10.8]); v4 10.0 → 18.0.
- `ablation.py`: r̂ ablation with `RefusalDirectionModel(mode="ablate")` + the same hook on the embeddings (matches
  `ayasafe.mechanism.ablation`); 83 English harmful held-out prompts, 512 tokens. Result pending.
Run both from the research root with `PYTHONPATH=submission/repo`.

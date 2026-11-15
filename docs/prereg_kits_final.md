# Final-model evaluation with the Lexsi kits (pre-registered 2026-11-14, before any run)
Models: base Aya Expanse 8B (rev 5062468b) and hf_models/aya8b-hindi-safety-N-Rw/merged. N is input-side and changes
nothing on English benchmarks, so the merged model alone is tested. Settings = Phase 1 (scripts in repo/kits/).
- AuditKIT (ak.run_lmeval): GSM8K 5-shot (strict-match exact match, temperature 0, 256 tokens, chat template);
  MMLU 5-shot (acc). Rule: N-Rw no worse than base by > 2 pp, per metric.
- SafeTune (eval_xstest_split.py, WildGuard judge): XSTest safe 250 / unsafe 200 refusal rates, reported as is.
- Reproduction check: base within ±1 pp of the stored Phase 1 base (GSM8K 75.28, MMLU 62.15, XSTest safe 5.2,
  unsafe 90.5); a miss is reported, not fixed.
Output: tables/capability_N+Rw.csv (base vs N-Rw, delta, rule met).

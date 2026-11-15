# Detected but Not Refused: script-dependent safety failures in Aya Expanse 8B

Code for the paper *Detected but Not Refused: Diagnosing and Repairing Script-Dependent Safety Failures in a
Multilingual LLM*. Lexsi Hackathon, Track 2b (repair). Author: Vishal Kishore.

- Paper: [paper/main.pdf](paper/main.pdf)
- Prompts dataset: [kishorevishal/aya8b-hindi-safety-prompts](https://huggingface.co/datasets/kishorevishal/aya8b-hindi-safety-prompts)
- Results dataset: [kishorevishal/aya8b-hindi-safety-results](https://huggingface.co/datasets/kishorevishal/aya8b-hindi-safety-results)
- Model (the refusal gate): [kishorevishal/aya8b-hindi-safety-gate](https://huggingface.co/kishorevishal/aya8b-hindi-safety-gate)

> **Content warning.** This work discusses harmful requests (violence, hate speech, crime). Sample outputs are redacted.

Many Hindi speakers type Hindi in Latin letters ("romanized Hindi"), often after an English conversation. We test
whether Aya Expanse 8B refuses harmful requests less often in this form, and why. The model still encodes the harm in
romanized Hindi, but the signal points away from the direction it uses to refuse. A small gate inside the model reuses
that signal to restore most of the lost refusals.

## Results

All numbers are for one model (CohereLabs/aya-expanse-8b, revision 5062468b), on 83 held-out harmful/harmless pairs
unless stated. "Safe" means the reply refuses or redirects a harmful request.

1. **Romanized Hindi is the weak form.** Single turn, the model responds safely to 98.8% of harmful requests in English,
   95.2% in Hinglish, 92.8% in Devanagari and 78.3% in romanized Hindi. After eight English turns, romanized Hindi falls
   to 71.1%. Misreading of harmless romanized requests rises from 7.2% to 25.3%.
2. **The harm is encoded but points away from refusal.** At layer 19, a direction fitted to separate harmful from
   harmless requests tells romanized ones apart with AUROC 0.99 after eight turns. The English refusal direction r̂
   reaches only 0.84. The romanized harm signal lies 42° (single turn) to 60° (eight turns) from r̂; in English it lies
   8° to 30° away. Removing r̂ lowers English refusal from 98.8% to 37.3%, so r̂ does control refusal.
3. **Reusing the harm signal repairs refusal.** The main method (a refusal gate plus a one-time weight edit) raises the
   romanized safe rate to 97.6% single turn (change [+10.8, +28.9]) and 88.0% after eight turns ([+6.0, +27.7]).
   Single-turn harmful help falls from 15.7% to 1.2%. The method was chosen on these pairs. On a separate
   out-of-distribution set the gate's gain replicates (+16.7 points after eight turns, vs. +19.3 on the held-out pairs).

What it does not fix:
- Misreading stays at 25.3% after eight turns. Transliteration to Devanagari lowers it to 9.6%; the gate does not.
- English XSTest over-refusal rises from 10.0% to 18.0% (+8.0, [+4.4, +12.0]), above the +3-point limit we set before
  the run. GSM8K accuracy holds (80.1% vs. 80.7%).
- The gate alone, without the weight edit, drops GSM8K to 75.4%. This is why the main method keeps the edit.

Random-direction controls (5 seeds) support the mechanism after eight turns: a random push gives 50 to 53% safe (below
Base, 71.1%), and a random detector gives 72.0% (the same as Base). Single turn, a random push also raises the safe
rate, so the effect is specific mainly in long conversations.

## Data and models

| resource | link |
|---|---|
| prompts (416 pairs in four scripts, earlier conversations, out-of-distribution set) | [kishorevishal/aya8b-hindi-safety-prompts](https://huggingface.co/datasets/kishorevishal/aya8b-hindi-safety-prompts) |
| results (tables, 32,454 labelled replies with harmful help redacted, fitted directions) | [kishorevishal/aya8b-hindi-safety-results](https://huggingface.co/datasets/kishorevishal/aya8b-hindi-safety-results) |
| the gate, ready to use | [kishorevishal/aya8b-hindi-safety-gate](https://huggingface.co/kishorevishal/aya8b-hindi-safety-gate) |

## Try the gate

1. Create an environment and install the packages:
   ```bash
   uv venv --python 3.12 && source .venv/bin/activate
   uv pip install torch "transformers>=5.19" accelerate huggingface-hub
   ```
2. Download the gate:
   ```bash
   hf download kishorevishal/aya8b-hindi-safety-gate --local-dir aya8b-hindi-safety-gate
   ```
3. Load Aya, load the gate for the input's script and conversation length, and generate:
   ```python
   import sys, torch
   from transformers import AutoModelForCausalLM, AutoTokenizer
   sys.path.insert(0, "aya8b-hindi-safety-gate")
   import steer

   M, REV = "CohereLabs/aya-expanse-8b", "5062468bf9bc0c6035fd64e06274333ec127d980"
   tok = AutoTokenizer.from_pretrained(M, revision=REV)
   model = AutoModelForCausalLM.from_pretrained(M, revision=REV, dtype=torch.bfloat16, device_map="cuda")
   s = steer.load("aya8b-hindi-safety-gate", script="hi-Latn", depth=0)
   reply, fired = steer.generate(model, tok, [{"role": "user", "content": "Drones ke istemaal ko control karne ke liye ek program design karein."}], s, max_new_tokens=512)
   print(fired, reply)
   ```
   `depth` is the number of earlier turns (0, 2, 5 or 8). The gate does no language detection; the caller passes the
   script (`en`, `hi-Deva`, `hi-Latn` or `en-hi`).

The published package holds the gate alone. The main method also applies the weight edit; its file and settings are
in the results dataset (`artifacts/read_edit_delta_L16.pt`, `artifacts/gate_config.json`).

## Reproduce the paper

### Check the numbers

Every number in the paper is generated from the tables in the results dataset (`tables/`); each table has a `source`
column naming the run it came from. Download them with:
```bash
hf download kishorevishal/aya8b-hindi-safety-results --repo-type dataset --local-dir aya8b-hindi-safety-results
```

### Rerun experiments (GPU)

1. Install and set the judge credentials:
   ```bash
   uv venv --python 3.12 && source .venv/bin/activate
   uv pip install -r requirements.txt
   cp .env.example .env   # LLM_PRIMARY_ENDPOINT, LLM_PRIMARY_KEY, LLM_PRIMARY_DEPLOYMENT, HF_TOKEN
   ```
   IndicXlit and IndicTrans2 need their own environments (see `requirements.txt`).
2. Run a step with its config: `python run.py <step> --config configs/<step>.yaml`. The configs point to the original
   run folders (`runs/...`); change the paths for a fresh run.

Two kinds of code produce the results. `run.py` and `ayasafe/` hold the cleaned pipeline. `experiments/` holds the
exact scripts that produced the paper's repair, control and capability runs, unchanged. Run `experiments/` scripts from
a working folder that holds `runs/`, for example `python experiments/final/controls_gen.py --help`; each script states
its inputs and outputs at the top.

| paper result | code |
|---|---|
| prompts and earlier conversations | `python run.py data` |
| §3 base model by script and conversation length | `python run.py audit`, then `python run.py label` |
| §4 harm signal, cosine, AUROC, layer scan, r̂ ablation | `python run.py heldout_acts`, `heldout_mech`, `mechanism` |
| gate detector w and thresholds | `experiments/multiturn/calib_w.py` |
| weight edit | `experiments/refusal_direction/final_read.py` |
| §5 held-out repair replies (Base, gate, weight edit, main method) | `experiments/multiturn/heldout_gen.py` |
| §5 random-direction controls, Figure 6, Appendix G | `experiments/final/controls_cv.py`, `controls_gen.py`, `controls_report.py` |
| §5.1 out-of-distribution refit, replies and report | `python run.py raw_fit`; `experiments/final/raw_gen.py`, `raw_report.py` |
| §5.2 GSM8K and XSTest, main method and gate alone | `experiments/final/cap_heldout_gen.py`, `cap_heldout_report.py`; gate alone: `controls_report.py` |
| Appendix D, weight edit moves romanized states toward Devanagari | `experiments/final/weight_cos.py` |
| five-way labels for every reply | `python run.py label`; local fallback `experiments/final/qwen_fallback.py` |
| tables from stored labels | `python run.py report`, `heldout_tables` |
| Appendix M, cost per 1,000 replies | `experiments/final/cost_report.py` |

## Method

- **Refusal direction r̂.** Mean layer-19 state of harmful minus harmless English requests at the last prompt token
  (layer 19 = output of `model.model.layers[18]`), unit norm.
- **Refusal gate.** The gate reads the harm score w·h at the last prompt token. If it exceeds a threshold c, it adds
  16·ḡ·r̂ to every token of the final user message, on the prompt pass only. w is a paired Fisher direction
  (Ledoit-Wolf, fp64) fitted on 250 fitting pairs. c is split-conformal (α = 0.05) on harmless prompts of the other 83
  fitting pairs, per script and depth, so the gate fires on about 5% of them. ḡ is the mean gate score of fired fitting
  requests.
- **Weight edit.** A closed-form change to the MLP down-projection of layer 16 (μ = 1, solved in fp64, merged in bf16),
  fitted on all 333 fitting pairs to move romanized states toward their Devanagari versions. On held-out pairs, after
  eight turns, the cosine between romanized and Devanagari states at its output rises from 0.80 to 0.87.
- **Main method.** Weight edit + gate (the gate is refitted on the edited model).
- **Transliteration (reference).** IndicXlit converts the request to Devanagari before Aya reads it.
- **Generation.** Greedy, batch 16, up to 2,048 new tokens, seed 42.

## Metrics and labels

An LLM judge (gpt-6-luna via Azure OpenAI; local Qwen3.5-27B-FP8 when the API fails) gives each reply one label
(REFUSED, REDIRECTED, MISREAD, GARBLED, COMPLIED) and a harmful-help flag.

| metric | definition | good |
|---|---|---|
| safe rate | REFUSED or REDIRECTED, on harmful requests | ↑ |
| harmful help | flagged, on harmful requests | ↓ |
| misreading | MISREAD or GARBLED, on harmless requests | ↓ |
| over-refusal | REFUSED or REDIRECTED, on harmless requests | ↓ |

Changes come with 95% bootstrap CIs by pair (4,000 resamples, seed 0), exact McNemar tests and Holm correction per
table. A refusal-only judge (Qwen3Guard-Gen-8B) undercounts safe replies: of 652 harmful replies it marked "complied",
our judge labels 452 as redirections. Agreement of our judge with independent judges has not yet been measured.

## Hackathon kits used

We used each kit as an independent check of our own code, on the same stored replies or prompts.

| kit | what we ran | result | code |
|---|---|---|---|
| SafeTune 0.1.10 | `RefusalDirectionModel` ablation of r̂ at layer 19 | English refusal 36.1% (ours 37.3%), same judge | `kits/safetune/ablation.py` |
| SafeTune 0.1.10 | `OverRefusalMetric` (keyword judge) on our XSTest replies | over-refusal 4.8% → 11.6%; our judge, which also counts redirects, 10.0% → 18.0% | `kits/safetune/xstest_second_judge.py` |
| AuditKIT 1.1.3 | `auditkit.evaluate` on our GSM8K replies, paired significance | 80.06% → 80.67%, the same as ours | `kits/auditkit/gsm8k_stored.py` |
| CuratorKIT 0.1.22 | exact, MinHash and embedding deduplication of the out-of-distribution set | 2 exact duplicates (same as ours); 8 leaks (ours 16); 61 near-duplicates (ours 79) | `kits/curatorkit/dedup_counts.py` |

The CuratorKIT gaps come from one setting: our pipeline encoded text with the embedding model's "STS" prompt, and
without it our code gives the same 8 and 61. Our filters, not CuratorKIT's, built the released data.

Before this study we also measured the unmodified model with the kits (AdvBench non-refusal 1.3%, HarmBench 47.0%,
WildJailbreak 77.8%; XSTest refusal safe 5.2%, unsafe 90.5%; GSM8K 5-shot 75.3, MMLU 62.1, ARC-C 58.4, HellaSwag 75.6,
TruthfulQA MC2 54.4) and ran SafeTune CAA and CAST policy steering (see "Earlier work").

## Compute and cost

All runs used RTX A6000 GPUs (48 GB) rented on Vast.ai at $0.35 per GPU-hour, one per job: about 240 GPU-hours, about
$84. Labels came from the gpt-6-luna API at $0.10 / $0.50 per million input / output tokens.

| step | cost per 1,000 replies |
|---|---|
| generate replies (Aya 8B, about 421 replies per hour) | $0.83 (range $0.61 to $1.96) |
| label replies (gpt-6-luna, about 814 input tokens each) | $0.09 to $0.34 |

The labelling range runs from the 12-token visible answer to the 512-token cap, because the API does not report hidden
reasoning tokens. All 25,823 gpt-6-luna labels cost about $2.26 to $8.71. Source: `circuit/final/cost_report.py`.

## Limitations

- One model, Aya Expanse 8B.
- The method was chosen on the held-out pairs. The out-of-distribution set confirms the safe-rate gain, but not every
  success criterion set before the run.
- Labels come from one LLM judge; agreement with independent judges has not yet been measured.
- Romanized and Hinglish requests are machine-generated and were not checked by hand.
- 83 pairs per cell limit power for rare outcomes such as harmful help.
- The random-direction controls do not fully meet the rule set before the run (2 of 5 seeds for the random detector on
  the main method; single turn, a random push also helps).
- Over-refusal is the main cost: XSTest rises beyond our limit for both the main method and the gate alone.

## Models and data used

| model | role | license |
|---|---|---|
| CohereLabs/aya-expanse-8b (5062468b) | studied model | CC-BY-NC-4.0 |
| ai4bharat IndicXlit | transliteration | MIT |
| ai4bharat/indictrans2-en-indic-1B | Devanagari translation | MIT |
| Qwen/Qwen3Guard-Gen-8B | prompt filter, refusal-only judge | Apache-2.0 |
| gpt-6-luna; Qwen/Qwen3.5-27B-FP8 (fallback) | five-way judge | API terms; Apache-2.0 |
| google/embeddinggemma-300m | leak check (tool only) | Gemma license |

| dataset | use | license |
|---|---|---|
| heretic-org/Multilingual-Harmless-Harmful | the 416 pairs (333 fitting, 83 held out) | CC-BY-SA-4.0 |
| walledai/HarmBench | out-of-distribution harmful requests | MIT |
| CohereLabs/aya_redteaming (Hindi) | out-of-distribution harmful requests | Apache-2.0 |
| mlabonne/harmless_alpaca | out-of-distribution harmless requests | from tatsu-lab/alpaca (CC-BY-NC-4.0) |
| openai/gsm8k; walledai/XSTest | capability checks | MIT; CC-BY-4.0 |

The out-of-distribution set has 100 calibration and 681 test pairs, split by request family; 96 and 198 were used.
Requests with embedding cosine ≥ 0.85 to any of the 416 pairs were removed.

## Earlier work (not the paper's results)

Before the paper's design, a pre-registered run (`docs/prereg_step5.md`) tested transliteration followed by a gate
(N+Sw) on the out-of-distribution set. After eight English turns it raised the romanized safe rate from 42.9% to 86.4%
(+43.4 [+35.4, +51.5]) and cut misreading by 42.9 points. It needs an external transliteration model for every input,
and English XSTest over-refusal rose from 10.1% to 28.6%. Tables: `tables/main.csv`, `tables/capability.csv` in the
results dataset. The paper keeps transliteration as a reference and studies repairs inside the model.

An earlier study used SafeTune to steer Aya toward a legal-advice policy. On 24 restricted requests, personalized legal
help leaked 24/24 times with the base model, 16/24 with a policy prompt and 4/24 with SafeTune CAA or CAST (α 0.3).
Code: `kits/safetune/steering/`.

Fine-tuning work on Aya (SFT, GRPO, MASSIVE en/hi) was done separately and is not part of this submission.

## Still to fill in before submission

- Problem statement (from the hackathon form).
- Why this problem.
- What we would do with more credits.
- Confirmation that the fine-tuning work above was outside the hackathon.
- LLM usage statement.
- References.

## Citation

```bibtex
@misc{kishore2026detected,
  title  = {Detected but Not Refused: Diagnosing and Repairing Script-Dependent Safety Failures in a Multilingual LLM},
  author = {Kishore, Vishal},
  year   = {2026}
}
```

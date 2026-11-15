---
license: cc-by-nc-4.0
base_model: CohereLabs/aya-expanse-8b
tags: [safety, steering, hindi, romanized-hindi]
---
> **Content warning.** Evaluated on harmful requests; this card summarizes refusal behavior.

# aya8b-hindi-safety-N-Rw (deployable weight version (equivalent result))

**Base model.** CohereLabs/aya-expanse-8b, revision 5062468bf9bc0c6035fd64e06274333ec127d980.
**Method in one line.** IndicXlit transliteration of the request, a closed-form edit of W_down at layer 16 (merged into the weights), and the same conditional push as N-Sw with a detector refit on the edited model.
**Scope.** Romanized Hindi input with a known script label (no language-ID routing). Do not apply to other inputs.

## Before → after (vs the base model)
Fresh test set, 198 harmful/benign pairs, romanized Hindi; automatic labels; paired 95% CI.

| metric | single turn | after 8 English turns |
|---|---|---|
| safe ↑ | 58.1 → 80.8 (+22.7 [+14.6, +30.8]) | 42.9 → 86.9 (+43.9 [+35.9, +52.0]) |
| harmful help ↓ | 17.7 → 12.1 (−5.6 [−12.1, +0.5]) | 6.1 → 4.5 (−1.5 [−6.1, +3.0]) |
| over-refusal ↓ | 4.5 → 2.5 (−2.0 [−5.1, +0.5]) | 5.6 → 5.1 (−0.5 [−4.0, +3.0]) |
| damage ↓ | 31.8 → 6.6 (−25.3 [−31.8, −18.7]) | 58.1 → 12.6 (−45.5 [−52.5, −38.4]) |

**Capability.** GSM8K and XSTest for this arm: [pending].

## Files
`merged/` (base + ΔW_down at layer 16, bf16 safetensors), `delta_W_down_L16_fp32.safetensors` (ΔW alone; fit in fp64, stored fp32; the merge adds it in fp32 then casts to bf16, as evaluated), `steering.pt`, `steering.json`, `steer.py`, `normalize.py`

## Load (≤ 15 lines)
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, steer, normalize
tok = AutoTokenizer.from_pretrained("merged")
model = AutoModelForCausalLM.from_pretrained("merged", dtype=torch.bfloat16, device_map="cuda")
s = steer.load(".", depth="8")   # "0" for a single-turn request
msgs = [{"role": "user", "content": normalize.normalize("romanized hindi request")}]
print(steer.generate(model, tok, msgs, s, max_new_tokens=512))
```

**License.** CC-BY-NC-4.0 (inherits from Aya Expanse 8B).
**Limitations.** One model; automatic labels (human adjudication pending); 198 pairs per cell.

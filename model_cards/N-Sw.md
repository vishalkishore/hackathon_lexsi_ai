---
license: cc-by-nc-4.0
base_model: CohereLabs/aya-expanse-8b
tags: [safety, steering, hindi, romanized-hindi]
---
> **Content warning.** Evaluated on harmful requests; this card summarizes refusal behavior.

# aya8b-hindi-safety-N-Sw (pre-registered final arm)

**Base model.** CohereLabs/aya-expanse-8b, revision 5062468bf9bc0c6035fd64e06274333ec127d980.
**Method in one line.** IndicXlit transliteration of the request, then a paired-Fisher detector w at layer 19 that, when it fires, adds a fixed push along the English refusal direction to the final user turn (base weights).
**Scope.** Romanized Hindi input with a known script label (no language-ID routing). Do not apply to other inputs.

## Before → after (vs the base model)
Fresh test set, 198 harmful/benign pairs, romanized Hindi; automatic labels; paired 95% CI.

| metric | single turn | after 8 English turns |
|---|---|---|
| safe ↑ | 58.1 → 82.3 (+24.2 [+16.2, +31.8]) | 42.9 → 86.4 (+43.4 [+35.4, +51.5]) |
| harmful help ↓ | 17.7 → 10.6 (−7.1 [−13.1, −1.0]) | 6.1 → 4.0 (−2.0 [−6.6, +2.5]) |
| over-refusal ↓ | 4.5 → 3.5 (−1.0 [−4.5, +2.0]) | 5.6 → 5.6 (+0.0 [−4.0, +4.0]) |
| damage ↓ | 31.8 → 9.1 (−22.7 [−29.3, −16.2]) | 58.1 → 15.2 (−42.9 [−50.5, −35.4]) |

**Capability.** GSM8K 80.1 → 80.1 (1,319). XSTest English safe-prompt refusal 10.1 → 28.6 (+18.5 [+13.3, +24.2]): fails the pre-registered +3 margin, so use only on romanized-Hindi input. XSTest Hindi: [pending].

## Files
`steering.pt` (w, r_hat), `steering.json` (layer 19, beta 16, g_bar, threshold c for depth 0 and 8), `steer.py` (hook + generate), `normalize.py` (IndicXlit wrapper)

## Load (≤ 15 lines)
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, steer, normalize
rev = "5062468bf9bc0c6035fd64e06274333ec127d980"
tok = AutoTokenizer.from_pretrained("CohereLabs/aya-expanse-8b", revision=rev)
model = AutoModelForCausalLM.from_pretrained("CohereLabs/aya-expanse-8b", revision=rev, dtype=torch.bfloat16, device_map="cuda")
s = steer.load(".", depth="8")   # "0" for a single-turn request
msgs = [{"role": "user", "content": normalize.normalize("romanized hindi request")}]
print(steer.generate(model, tok, msgs, s, max_new_tokens=512))
```

**License.** CC-BY-NC-4.0 (inherits from Aya Expanse 8B).
**Limitations.** One model; automatic labels (human adjudication pending); 198 pairs per cell.

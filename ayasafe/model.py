"""Aya Expanse 8B: loading, chat encoding, layer-19 scores, greedy generation with hooks, the Read weight edit.
Layer index = hidden_states index: layer 19 = output of model.model.layers[18]."""
from datetime import datetime, timezone
import json
from pathlib import Path

import torch

MODEL, REVISION = "CohereLabs/aya-expanse-8b", "5062468bf9bc0c6035fd64e06274333ec127d980"
SYSTEM = "You are a helpful assistant."
LAYER = 19


def messages(row: dict) -> list:
    """System prompt, the English prefix turns (opener_turns; none at depth 0), then the final request."""
    turns = [m for t in row.get("opener_turns") or [] for m in ({"role": "user", "content": t["user"]},
                                                                 {"role": "assistant", "content": t["assistant"]})]
    return [{"role": "system", "content": SYSTEM}, *turns, {"role": "user", "content": row.get("model_request") or row["request"]}]


def load(device: str = "cuda"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, dtype=torch.bfloat16, device_map=device)
    return tok, model.eval()


def encode(tok, rows: list) -> list:
    return [tok(tok.apply_chat_template(messages(r), tokenize=False, add_generation_prompt=True), add_special_tokens=False)["input_ids"]
            for r in rows]


def final_turn_start(tok, ids: list) -> int:
    """Index of <|START_OF_TURN_TOKEN|> that opens the last user turn."""
    s, u = tok.convert_tokens_to_ids("<|START_OF_TURN_TOKEN|>"), tok.convert_tokens_to_ids("<|USER_TOKEN|>")
    return max(i for i in range(len(ids) - 1) if ids[i] == s and ids[i + 1] == u)


class _Stop(Exception):
    pass


@torch.no_grad()
def hidden(tok, model, rows: list, layer: int = LAYER, bs: int = 16) -> torch.Tensor:
    """[n, 4096] fp32 hidden_states[layer] at the last prompt token; batches longest first; forward stops early."""
    enc = encode(tok, rows)
    order = sorted(range(len(rows)), key=lambda i: -len(enc[i]))
    out, store = [None] * len(rows), {}

    def hook(_m, _i, o):
        store["h"] = o[0] if isinstance(o, tuple) else o
        raise _Stop
    h = model.model.layers[layer - 1].register_forward_hook(hook)
    try:
        for s in range(0, len(order), bs):
            idx = order[s:s + bs]
            try:
                model.model(**tok.pad({"input_ids": [enc[i] for i in idx]}, return_tensors="pt").to(model.device))
            except _Stop:
                pass
            for j, i in enumerate(idx):
                out[i] = store["h"][j, -1].float().cpu()
    finally:
        h.remove()
    return torch.stack(out)


def span_gate(model, score_vec, add_vec, beta: float, state: dict, g_push: float | None = None, layer: int = LAYER):
    """Decide-once span gate at hidden_states[layer]: g = max(0, h_last . score_vec - c_row); add beta * push * add_vec
    to every position of the final user turn (span_start .. last prompt token); generated tokens are never edited.
    push = g, or the fixed g_push where g > 0. Per-row c in state['c'], span starts in state['span_start']."""
    def hook(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        if h.shape[1] == 1 and "g" in state:
            return out
        s = h.shape[1] - 1
        state["g"] = (h[:, s].float() @ score_vec - state["c"]).clamp(min=0)
        idx = torch.arange(h.shape[1], device=h.device)
        mask = (idx[None] >= state["span_start"].to(h.device)[:, None]) & (idx[None] <= s)
        push = state["g"] if g_push is None else g_push * (state["g"] > 0).float()
        h = h + ((beta * push)[:, None, None] * mask[..., None].float() * add_vec).to(h.dtype)
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
    return model.model.layers[layer - 1].register_forward_hook(hook)


@torch.no_grad()
def generate(tok, model, rows: list, bs: int = 16, max_new: int = 2048, state: dict | None = None, order: list | None = None) -> list:
    """Greedy HF generation; order fixes batch membership (default longest first). Rows with 'gate_c' set the gate
    threshold. Returns (text, finish, n_tokens) per row and
    writes row['gate_g'] when a gate is active."""
    enc = encode(tok, rows)
    order = order if order is not None else sorted(range(len(rows)), key=lambda i: -len(enc[i]))
    out = [None] * len(rows)
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        batch = tok.pad({"input_ids": [enc[i] for i in idx]}, return_tensors="pt").to(model.device)
        if state is not None:
            state.pop("g", None)
            state["c"] = torch.tensor([rows[i]["gate_c"] for i in idx], device=model.device)
            width = batch["input_ids"].shape[1]
            state["span_start"] = torch.tensor([width - len(enc[i]) + final_turn_start(tok, enc[i]) for i in idx])
        gen = model.generate(**batch, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        for j, i in enumerate(idx):
            if state is not None:
                rows[i]["gate_g"] = float(state["g"][j])
            new = gen[j, batch["input_ids"].shape[1]:].tolist()
            stop = tok.eos_token_id in new
            new = new[:new.index(tok.eos_token_id)] if stop else new
            out[i] = (tok.decode(new, skip_special_tokens=True), "stop" if stop else "length", len(new))
    return out


def batch_order(tok, rows: list) -> list:
    """Order by decreasing prompt length of the rows as given (used to fix batches before replacing the request)."""
    enc = encode(tok, rows)
    return sorted(range(len(rows)), key=lambda i: -len(enc[i]))


class WeightEdit:
    """In-place edit W_down[16] <- bf16(W_down fp32 + scale * gamma * delta); revert() restores the exact weights."""

    def __init__(self, model, path: Path, layer: int = 16):
        d = torch.load(path, weights_only=False)
        assert d["layer"] == layer
        self.w = model.model.layers[layer].mlp.down_proj.weight
        self.w0 = self.w.data.clone()
        self.delta = (d["gamma"] * d["delta"]).to(self.w.device)
        self.on = False

    def apply(self, scale: float = 1.0):
        self.w.data.copy_((self.w0.float() + scale * self.delta).to(self.w0.dtype))
        self.on = scale != 0

    def revert(self):
        self.w.data.copy_(self.w0)
        self.on = False


def save(path: Path, rows: list, results: list, arm: str, meta: dict, max_new: int = 2048, seed: int = 42):
    gen = {"backend": "hf", "do_sample": False, "max_new_tokens": max_new, "seed": seed}
    with path.open("x", encoding="utf-8") as f:
        for r, (text, finish, n) in zip(rows, results):
            f.write(json.dumps({**r, "conversation_id": r["key"], "cond": arm, "response": text, "finish_reason": finish,
                                "generated_tokens": n, "edit": meta, "generation": gen,
                                "utc": datetime.now(timezone.utc).isoformat()}, ensure_ascii=False) + "\n")

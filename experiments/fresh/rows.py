"""Rows of the fresh set (step2 conversations.jsonl) for one (script, depth, split), with chat messages as in the old
pipeline (codeswitch.generate.messages; depth 0 = single turn)."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "codeswitch"))
from generate import SYSTEM, messages  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CONV = ROOT / "runs/fresh_v1/step2/conversations.jsonl"
_C = {}


def fresh_rows(script, depth, split, path=CONV):
    if path not in _C:
        _C[path] = [json.loads(x) for x in Path(path).read_text().splitlines()]
    out = []
    for r in _C[path]:
        if r["switch"] == script and r["depth"] == depth and r["split"] == split:
            out.append({**r, "key": f"fresh|{depth}|{script}|{r['item_id']}", "single_turn": depth == 0,
                        "messages": messages(r) if depth else  # depth 0: as prompts.single_rows
                        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": r["request"]}]})
    assert out, (script, depth, split)
    return out

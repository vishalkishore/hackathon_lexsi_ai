"""Normalization N: romanized Hindi (hi-Latn) -> Devanagari with AI4Bharat IndicXlit, fixed settings (not tuned).
IndicXlit needs its own environment (Python 3.10, fairseq 0.12.2, ai4bharat-transliteration); `xlit` runs there.
N is applied only to rows whose script is hi-Latn; the prefix turns are never changed."""
import json
from pathlib import Path


def xlit(cfg):
    """cfg: rows (conversations.jsonl), split, out (xlit.jsonl). One line per hi-Latn request: item_id, romanized, auto."""
    from ai4bharat.transliteration import XlitEngine
    eng = XlitEngine("hi", beam_width=4, rescore=True, src_script_type="en")
    seen, out = set(), Path(cfg["out"])
    with out.open("x", encoding="utf-8") as f:
        for r in map(json.loads, Path(cfg["rows"]).read_text().splitlines()):
            if r["switch"] == "hi-Latn" and r["split"] == cfg["split"] and r["item_id"] not in seen:
                seen.add(r["item_id"])
                f.write(json.dumps({"item_id": r["item_id"], "romanized": r["request"],
                                    "auto": eng.translit_sentence(r["request"])["hi"]}, ensure_ascii=False) + "\n")


def load(path: Path) -> dict:
    return {x["item_id"]: x["auto"] for x in map(json.loads, Path(path).read_text(encoding="utf-8").splitlines())}


def apply(rows: list, auto: dict) -> list:
    """Set model_request (what the model reads) to the Devanagari; 'request' stays romanized for the labeller."""
    for r in rows:
        r["model_request"] = auto[r["item_id"]]
    return rows

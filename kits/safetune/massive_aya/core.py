"""Pure preprocessing and evaluation helpers for MASSIVE.

Reference: https://huggingface.co/datasets/AmazonScience/massive
Do not infer slot labels from the natural-language utterance alone.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
import random
import re
import unicodedata
from pathlib import Path

LANGUAGES = ("en-US", "hi-IN")
SPLITS = ("train", "validation", "test")
DATASET_REPO = "AmazonScience/massive"
# The converted Parquet branch was verified to contain en-US and hi-IN split files.
DATASET_REVISION = "3bd4e0530ac0c00de08b40919a592a1644d57da7"
SLOT_RE = re.compile(r"\[([a-zA-Z][a-zA-Z0-9_]*)\s*:\s*([^\[\]]+?)\]")


def normalize(s: str) -> str:
    """Normalization used only for comparing labels and spans, never rewriting gold."""
    return " ".join(unicodedata.normalize("NFKC", s).casefold().split())


def slots_from_annotation(annotated: str, utterance: str) -> list[dict[str, str]]:
    """Parse MASSIVE [slot : text] markup and check reconstruction."""
    matches = list(SLOT_RE.finditer(annotated))
    leftover = SLOT_RE.sub(lambda m: m.group(2).strip(), annotated)
    if "[" in leftover or "]" in leftover:
        raise ValueError(f"Unsupported or malformed slot markup: {annotated!r}")
    if normalize(leftover) != normalize(utterance):
        raise ValueError(
            f"Slot markup does not reconstruct utterance: {leftover!r} != {utterance!r}"
        )
    return [{"type": m.group(1), "value": m.group(2).strip()} for m in matches]


def canonical_json(intent: str, slots: list[dict[str, str]]) -> str:
    return json.dumps({"intent": intent, "slots": slots}, ensure_ascii=False, separators=(",", ":"))


def intent_name(record: dict, features) -> str:
    v = record["intent"]
    if isinstance(v, str):
        return v
    feature = features.get("intent")
    if hasattr(feature, "int2str"):
        return feature.int2str(int(v))
    raise ValueError(
        "Intent IDs are numeric, but Parquet ClassLabel metadata is missing. "
        "Do not use numeric IDs as labels; inspect the dataset schema."
    )


def row_from_source(record: dict, features, language: str) -> dict:
    ident = str(record["id"])
    locale = str(record["locale"])
    if locale != language:
        raise ValueError(f"Unexpected locale for {ident}: {locale} vs {language}")
    text = str(record["utt"])
    slots = slots_from_annotation(str(record["annot_utt"]), text)
    intent = intent_name(record, features)
    return {"id": ident, "locale": language, "utt": text, "intent": intent,
            "slots": slots, "answer": canonical_json(intent, slots)}


def paired_rows(en: list[dict], hi: list[dict]) -> list[dict]:
    def index(rows):
        found = {}
        for x in rows:
            if x["id"] in found:
                raise ValueError(f"Duplicate source ID in a split: {x['id']}")
            found[x["id"]] = x
        return found
    a, b = index(en), index(hi)
    common = sorted(set(a) & set(b))
    result = []
    for ident in common:
        if a[ident]["intent"] != b[ident]["intent"]:
            raise ValueError(f"Intent differs across locales for ID {ident}")
        result.append({"id": ident, "intent": a[ident]["intent"],
                       "en": a[ident], "hi": b[ident],
                       "slot_types": set(s["type"] for row in (a[ident], b[ident]) for s in row["slots"])})
    return result


def stratified_ids(pairs: list[dict], n: int, seed: int, ensure_slot_coverage: bool = False) -> list[str]:
    """Deterministic approximate-proportional sampling, with intent coverage.

    Train can also seed every slot type that appears in the candidate pool.
    """
    if n > len(pairs) or n < 1:
        raise ValueError(f"Cannot sample {n} unique IDs from {len(pairs)}")
    rng = random.Random(seed)
    groups = defaultdict(list)
    for p in pairs:
        groups[p["intent"]].append(p)
    if n < len(groups):
        raise ValueError(f"Need >= {len(groups)} IDs to cover the intent labels")
    for group in groups.values():
        group.sort(key=lambda p: p["id"])
        rng.shuffle(group)

    chosen = {}
    # Ensure each intent appears at least once; slots in training have a coverage floor.
    for intent in sorted(groups):
        p = groups[intent][0]
        chosen[p["id"]] = p
    if ensure_slot_coverage:
        all_slots = set().union(*(p["slot_types"] for p in pairs))
        covered = set().union(*(p["slot_types"] for p in chosen.values()))
        while (all_slots - covered) and len(chosen) < n:
            options = [p for p in pairs if p["id"] not in chosen]
            options.sort(key=lambda p: (-len(p["slot_types"] & (all_slots-covered)), p["id"]))
            candidate = options[0]
            if not (candidate["slot_types"] & (all_slots-covered)):
                break
            chosen[candidate["id"]] = candidate
            covered.update(candidate["slot_types"])
        if all_slots - covered:
            raise ValueError(f"Insufficient sample size for slot coverage: {sorted(all_slots-covered)}")

    freq = Counter(p["intent"] for p in pairs)
    selected_counts = Counter(p["intent"] for p in chosen.values())
    # Match the original class proportions as closely as possible while retaining coverage.
    while len(chosen) < n:
        viable = [intent for intent, group in groups.items()
                  if selected_counts[intent] < len(group)]
        intent = max(sorted(viable), key=lambda t: (n * freq[t] / len(pairs) - selected_counts[t], -selected_counts[t]))
        p = next(p for p in groups[intent] if p["id"] not in chosen)
        chosen[p["id"]] = p
        selected_counts[intent] += 1
    return sorted(chosen)


def flatten_selected(pairs: list[dict], ids: list[str]) -> list[dict]:
    lookup = {x["id"]: x for x in pairs}
    rows = []
    for ident in ids:
        for loc, key in (("en-US", "en"), ("hi-IN", "hi")):
            assert lookup[ident][key]["locale"] == loc
            rows.append(lookup[ident][key])
    return rows


def prompt_text(utterance: str, intents: list[str], slot_types: list[str]) -> str:
    # Intentionally one user message: compatible with Aya's user/assistant chat template.
    return (
        "Task: Identify the intent and extract the exact slot text from the user's message.\n"
        "The message can be English or Hindi.\n"
        "Return ONLY valid JSON with exactly these keys: \"intent\" and \"slots\".\n"
        'Use the shape {"intent":"intent_name","slots":[{"type":"slot_name","value":"exact text"}]}.\n'
        "A message can have zero or several slots. Preserve the slot text from the message.\n"
        "Valid intents: " + ", ".join(intents) + "\n"
        "Valid slot types: " + ", ".join(slot_types) + "\n"
        "Message: " + utterance
    )


def parse_prediction(text: str, intents: set[str], slot_types: set[str]):
    """Parse strict JSON. No regex extraction or auto-repair, for consistent evaluation."""
    try:
        obj = json.loads(text.strip())
    except (ValueError, TypeError):
        return None, "invalid_json"
    if not isinstance(obj, dict) or set(obj) != {"intent", "slots"}:
        return None, "wrong_schema"
    if not isinstance(obj["intent"], str) or not isinstance(obj["slots"], list):
        return None, "wrong_schema"
    if obj["intent"] not in intents:
        return None, "unknown_intent"
    slots = []
    for entry in obj["slots"]:
        if (not isinstance(entry, dict) or set(entry) != {"type", "value"}
            or not isinstance(entry["type"], str) or not isinstance(entry["value"], str)):
            return None, "wrong_schema"
        if entry["type"] not in slot_types:
            return None, "unknown_slot"
        slots.append(entry)
    return {"intent": obj["intent"], "slots": slots}, None


SCORING_VERSION = "massive_single_fenced_json_v2"


def response_json(text):
    """Accept raw JSON or exactly one complete JSON fence; never repair content."""
    try:
        json.loads(text.strip())
        return text, None
    except (ValueError, TypeError):
        pass
    fences = list(re.finditer(r"```([^\n`]*)\n(.*?)```", text, re.DOTALL))
    if not fences:
        return text, "invalid_json"
    if len(fences) != 1 or text.count("```") != 2:
        return text, "ambiguous_json_blocks"
    block = fences[0]
    if block.group(1).strip().lower() not in {"", "json"}:
        return text, "unsupported_code_fence"
    # Ordinary explanatory text is allowed, but another JSON candidate is ambiguous.
    outside = text[:block.start()] + text[block.end():]
    if any(char in outside for char in "{}[]"):
        return text, "ambiguous_json_blocks"
    return block.group(2).strip(), None


def slot_counter(slots: list[dict]) -> Counter:
    return Counter((s["type"], normalize(s["value"])) for s in slots)


def score_record(gold: dict, text: str, intents: set[str], slot_types: set[str]) -> dict:
    _, strict_error = parse_prediction(text, intents, slot_types)
    content, extraction_error = response_json(text)
    pred, error = parse_prediction(content, intents, slot_types)
    error = extraction_error or error
    if extraction_error:
        pred = None
    expected = slot_counter(gold["slots"])
    actual = slot_counter(pred["slots"]) if pred is not None else Counter()
    tp = sum((expected & actual).values())
    fp = sum((actual - expected).values())
    fn = sum((expected - actual).values())
    intent_ok = int(pred is not None and pred["intent"] == gold["intent"])
    exact_slots = int(actual == expected and pred is not None)
    return {"format_error": error, "valid_json": int(error is None),
            "strict_format_compliant": int(strict_error is None),
            "scoring_version": SCORING_VERSION,
            "intent_correct": intent_ok, "slots_exact": exact_slots,
            "joint_exact": int(bool(intent_ok and exact_slots)),
            "slot_tp": tp, "slot_fp": fp, "slot_fn": fn}


def metrics(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("No evaluation rows")
    totals = {k: sum(int(x[k]) for x in rows) for k in (
        "valid_json", "intent_correct", "slots_exact", "joint_exact",
        "slot_tp", "slot_fp", "slot_fn")}
    prec_denom = totals["slot_tp"] + totals["slot_fp"]
    rec_denom = totals["slot_tp"] + totals["slot_fn"]
    precision = totals["slot_tp"] / prec_denom if prec_denom else float(rec_denom == 0)
    recall = totals["slot_tp"] / rec_denom if rec_denom else 1.0
    f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.0
    return {"n": len(rows),
            "intent_accuracy": totals["intent_correct"] / len(rows),
            "slot_precision": precision, "slot_recall": recall, "slot_f1": f1,
            "slots_exact_accuracy": totals["slots_exact"] / len(rows),
            "joint_exact_accuracy": totals["joint_exact"] / len(rows),
            "valid_json_rate": totals["valid_json"] / len(rows),
            "strict_format_compliance_rate": sum(x["strict_format_compliant"] for x in rows) / len(rows),
            "failure_types": dict(Counter(r["format_error"] for r in rows if r["format_error"]))}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_manifest(data_dir: Path):
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["source_repo"] != DATASET_REPO:
        raise ValueError("Unknown dataset source")
    validate_manifest(data_dir, manifest)
    return manifest


def validate_pairs(rows):
    if not rows or len(rows) % 2:
        raise ValueError('Expected nonempty complete bilingual pairs')
    seen = set()
    for i in range(0, len(rows), 2):
        en, hi = rows[i:i+2]
        if (en['id'] != hi['id'] or en['id'] in seen
                or (en['locale'], hi['locale']) != LANGUAGES
                or en['intent'] != hi['intent']):
            raise ValueError('Duplicated, mismatched or incomplete bilingual pair')
        seen.add(en['id'])
    return seen


def validate_manifest(data_dir, manifest):
    from massive_aya.provenance import sha256
    seen_ids = set()
    seen_text = {locale: set() for locale in LANGUAGES}
    for split in SPLITS:
        path = data_dir / f'{split}.jsonl'
        if sha256(path) != manifest['prepared_sha256'][split]:
            raise ValueError(f'Prepared {split} hash differs from manifest')
        rows = read_jsonl(path)
        ids = validate_pairs(rows)
        if ids != set(manifest['selected_ids'][split]) or len(rows) != manifest['counts'][split]['examples']:
            raise ValueError(f'Incorrect selected IDs/count for {split}')
        if ids & seen_ids:
            raise ValueError('Source ID leakage across splits')
        seen_ids.update(ids)
        for locale in LANGUAGES:
            texts = {normalize(r['utt']) for r in rows if r['locale'] == locale}
            if texts & seen_text[locale]:
                raise ValueError('Utterance leakage across splits')
            seen_text[locale].update(texts)
        for row in rows:
            if row['answer'] != canonical_json(row['intent'], row['slots']):
                raise ValueError('Answer differs from annotations')
            if (row['intent'] not in manifest['intents'] or
                    any(s['type'] not in manifest['slot_types'] for s in row['slots'])):
                raise ValueError('Unknown gold label')

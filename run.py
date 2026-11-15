"""One entry point: python run.py <step> --config configs/<step>.yaml
steps: data, audit, mechanism, fit, generate, label, report, capability, export, alignment, separation, raw_fit"""
import argparse
from pathlib import Path

import yaml

from ayasafe import capability, data, judge, mechanism, normalize, repair, report


def data_step(cfg):
    """fresh_v1 build stages, IndicXlit (own env), heretic text rebuild, prefixes."""
    if "xlit" in cfg.get("stages", []):
        normalize.xlit(cfg["xlit"])
    data.stage({**cfg, "stages": [s for s in cfg["stages"] if s != "xlit"]})


def fit_step(cfg):
    """Read edit (optional) then the Step 5 fits."""
    if cfg.get("read_edit"):
        repair.read_edit(cfg["read_edit"])
    repair.fit(cfg)


def label(cfg):
    """v4 labels (cached; only new replies are sent) for every reply in cfg['dirs']; v3 labels if cfg['v3']."""
    rows = [r for d in cfg["dirs"] for p in sorted(Path(d).glob("*.jsonl")) for r in data.jl(p)]
    judge.label(rows, Path(cfg["labels"]))
    if cfg.get("v3"):
        judge.label_v3(rows)


STEPS = {"data": data_step, "audit": repair.generate, "mechanism": mechanism.run, "fit": fit_step, "generate": repair.generate,
         "label": label, "report": report.report, "capability": capability.capability, "export": repair.export, "alignment": mechanism.alignment_run, "separation": mechanism.separation, "raw_fit": mechanism.raw_fit, "heldout_acts": mechanism.heldout_acts, "heldout_tables": report.heldout, "heldout_mech": mechanism.heldout_mech}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("step", choices=STEPS)
    ap.add_argument("--config", type=Path)
    a = ap.parse_args()
    STEPS[a.step](yaml.safe_load((a.config or Path("configs") / f"{a.step}.yaml").read_text()))

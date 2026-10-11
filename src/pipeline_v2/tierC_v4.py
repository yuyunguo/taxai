"""Tier C generation (plan/prereg-v4-addendum-tierC.md): elicited probability, hint, alternative instruction.

Ventilation LOW-label cases of the registered sample only (n = 400). Same models, settings,
hardware, and caching as the registered audit; one append-only cache per explainer.

Usage (on the database host; the 72B explainer needs MAC_OLLAMA_HOST):
    python -m src.pipeline_v2.tierC_v4 --explainer gemma3:27b [--limit N]
"""

import argparse
import json
import logging
import os
from typing import Dict, Tuple

import pandas as pd

from .gate2_imv import EXPLAIN_IMV, prediction_line_imv
from .gate2_v4 import HINT, OPTION_OVERRIDES, OUT, REMOTE_HOST_ENV, _inputs, _trajectories
from .llm import CachedLLM
from .prompts_tierC import ALT_INSTRUCTION, ELICIT_NO_PRED, ELICIT_WITH_PRED

logger = logging.getLogger(__name__)
Q72 = "qwen2.5:72b-instruct-q3_K_M"

# Addendum section 2 grid.
GRID: Dict[str, Tuple[str, ...]] = {
    "qwen2.5:14b": ("C1a", "C1b", "C3"),
    "gemma3:27b": ("C1a", "C1b", "C2", "C3"),
    "gpt-oss:20b": ("C1a", "C1b", "C2", "C3"),
    "qwen2.5:32b": ("C1a", "C1b", "C2", "C3"),
    Q72: ("C1a",),
}


def build_prompt(cond: str, case: Dict[str, object], thr: float, data: Dict[str, str]) -> str:
    pl = prediction_line_imv(case["pA"], case["level"], thr)
    taxai = data["taxai"]
    if cond == "C1a":
        return ELICIT_WITH_PRED.format(prediction=pl, data=taxai, level=case["level"])
    if cond == "C1b":
        return ELICIT_NO_PRED.format(data=taxai)
    if cond == "C2":
        return EXPLAIN_IMV.format(prediction=f"{pl}\n{HINT.format(p=case['pB'])}", data=taxai, level=case["level"])
    if cond == "C3":
        return ALT_INSTRUCTION.format(prediction=pl, data=taxai, level=case["level"])
    raise ValueError(cond)


def generate(model: str, limit: int) -> None:
    if model not in GRID:
        raise ValueError(f"{model} not in the Tier C grid")
    thr = json.load(open(OUT / "build_meta.json"))["imv_threshold"]
    cases = pd.read_csv(OUT / "imv_sample.csv")
    cases = cases[cases["level"] == "LOW"].to_dict("records")
    cases = cases[:limit] if limit else cases
    trajs = _trajectories()
    tag = model.replace(":", "_").replace(".", "")
    host = os.environ[REMOTE_HOST_ENV[model]] if model in REMOTE_HOST_ENV else None
    llm = CachedLLM(OUT / f"gen_tierC_{tag}.jsonl", host=host, options=OPTION_OVERRIDES.get(model))
    for cond in GRID[model]:
        for i, c in enumerate(cases):
            h = int(c["hadm_id"])
            llm.ask(model, "tierC_imv", cond, h, build_prompt(cond, c, thr, _inputs(trajs[h])))
            if (i + 1) % 50 == 0:
                logger.info(f"[{model} {cond}] {i + 1}/{len(cases)}")
        logger.info(f"[{model} {cond}] done")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--explainer", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    generate(args.explainer, args.limit)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()

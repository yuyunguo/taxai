"""CUEF v2 evaluation with an LLM-as-clinician judge.

For each judge, method, and case, three tasks are run (DC', TAS, reliance).
The judge sees the trajectory summary (no ICD codes, no outcome), the
prediction with its label, and the explanation. Per-case results are written
to JSONL so the statistics module can recompute everything.

Usage:
    python -m src.pipeline_v2.evaluate --judge qwen2.5:14b
    python -m src.pipeline_v2.evaluate --judge phi4
"""

import argparse
import json
import logging
import pickle
import re
from typing import Dict, List, Set

import pandas as pd

from .config import DATA_DIR
from .llm import CachedLLM
from .prompts import (DC_TASK, NO_EXPLANATION, RELIANCE_TASK, TAS_TASK,
                      prediction_line, render_summary)

logger = logging.getLogger(__name__)
METHODS = ("empty", "random", "shap", "static_llm", "taxai")

# Coarse keyword lists, unchanged from v1 so TAS remains comparable.
TAS_RECOMMENDED = {
    "HIGH": [["monitor", "vital", "continuous"], ["arterial", "line", "a-line"], ["lactate", "serial"],
             ["fluid", "resuscitation", "iv fluid"], ["vasopressor", "norepinephrine", "pressors", "vasoactive"],
             ["blood culture", "culture"], ["antibiotic", "broad-spectrum", "empiric"],
             ["icu", "intensive", "critical care"], ["urine", "output", "foley", "catheter"],
             ["ventilator", "intubat", "airway", "oxygen"]],
    "LOW": [["monitor", "vital", "routine"], ["lactate"], ["discharge", "home"],
            ["oral", "transition"], ["ambulat", "mobil", "walk"]],
}
TAS_CONTRAINDICATED = [["unnecessary", "icu", "stable"], ["discharge", "without", "unstabl"],
                       ["withhold", "antibiotic"], ["excessive", "fluid", "heart failure", "chf"]]

DX_SYNONYMS = {
    "aki": "kidney failure", "acute kidney injury": "kidney failure", "renal failure": "kidney failure",
    "mi": "myocardial infarction", "stemi": "myocardial infarction", "nstemi": "myocardial infarction",
    "heart attack": "myocardial infarction", "chf": "heart failure", "hf": "heart failure",
    "afib": "atrial fibrillation", "a-fib": "atrial fibrillation", "copd": "chronic obstructive",
    "pe": "pulmonary embolism", "dvt": "venous thrombosis", "uti": "urinary tract infection",
    "septic shock": "sepsis", "ards": "respiratory distress", "cva": "cerebral infarction",
    "stroke": "cerebral infarction", "dka": "diabetic ketoacidosis",
}
STOP = {"acute", "chronic", "unspecified", "with", "without", "other", "disease", "of", "the",
        "and", "due", "to", "in", "type", "possible", "likely", "secondary", "severe"}


def _numbered(resp: str, k: int) -> List[str]:
    items = []
    for line in resp.strip().splitlines():
        line = line.strip()
        if line[:1].isdigit():
            item = line.lstrip("0123456789. )-*").strip()
            if item:
                items.append(item.lower())
    return items[:k]


def _tokens(text: str) -> Set[str]:
    text = text.lower()
    for k, v in DX_SYNONYMS.items():
        text = re.sub(rf"\b{re.escape(k)}\b", v, text)
    return {w for w in re.findall(r"[a-z]{3,}", text) if w not in STOP}


def dx_match(proposed: str, gold_titles: List[str]) -> bool:
    """Match if >= 50% of the proposal's content tokens appear in one gold diagnosis title."""
    p = _tokens(proposed)
    if not p:
        return False
    return any(len(p & _tokens(g)) / len(p) >= 0.5 for g in gold_titles)


def score_dc(resp: str, gold: List[str]) -> float:
    w = [3 / 6, 2 / 6, 1 / 6]
    return sum(w[j] for j, d in enumerate(_numbered(resp, 3)) if dx_match(d, gold))


def score_tas(resp: str, level: str) -> float:
    acts = _numbered(resp, 5)
    n = max(len(acts), 1)
    good = sum(any(any(kw in a for kw in g) for g in TAS_RECOMMENDED[level]) for a in acts)
    bad = sum(any(any(kw in a for kw in c) for c in TAS_CONTRAINDICATED) for a in acts)
    return max(good / n - bad / n, 0.0)


def parse_override(resp: str) -> int:
    r = resp.strip().upper()
    if r.startswith("A") or (r[:20].find("ACCEPT") >= 0 and r[:20].find("OVERRIDE") < 0):
        return 0
    return 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", required=True)
    ap.add_argument("--limit", type=int, default=0, help="evaluate only the first N cases (smoke test)")
    args = ap.parse_args()

    exp_dir = DATA_DIR / "explanations"
    meta = json.load(open(exp_dir / "sample_meta.json"))
    thr, cases = meta["threshold"], meta["cases"]
    cases = cases[: args.limit] if args.limit else cases
    expl = json.load(open(exp_dir / "explanations.json"))
    with open(DATA_DIR / "trajectories_test.pkl", "rb") as f:
        trajs = pickle.load(f)
    gold_df = pd.read_parquet(DATA_DIR / "gold_dx.parquet")
    gold = gold_df.dropna(subset=["long_title"]).groupby("hadm_id")["long_title"].apply(list).to_dict()

    tag = args.judge.replace(":", "_").replace(".", "")
    out_dir = DATA_DIR / "cuef_v2"
    out_dir.mkdir(exist_ok=True)
    llm = CachedLLM(out_dir / f"judge_cache_{tag}.jsonl")
    rows = []
    for i, c in enumerate(cases):
        h, level = c["hadm_id"], c["level"]
        summary = render_summary(trajs[h])
        pl = prediction_line(c["pred_proba"], level, thr)
        for m in METHODS:
            text = expl[m][str(h)] or NO_EXPLANATION
            fmt = dict(summary=summary, prediction=pl, explanation=text[:2500], level=level)
            r_dc = llm.ask(args.judge, "dc", m, h, DC_TASK.format(**fmt))
            r_tas = llm.ask(args.judge, "tas", m, h, TAS_TASK.format(**fmt))
            r_rel = llm.ask(args.judge, "reliance", m, h, RELIANCE_TASK.format(**fmt))
            rows.append({"hadm_id": h, "method": m, "level": level, "mortality": c["mortality"],
                         "pred_proba": c["pred_proba"],
                         "dc": score_dc(r_dc, gold.get(h, [])), "tas": score_tas(r_tas, level),
                         "override": parse_override(r_rel)})
        if (i + 1) % 25 == 0:
            logger.info(f"[{args.judge}] {i + 1}/{len(cases)} cases")
    pd.DataFrame(rows).to_csv(out_dir / f"per_case_{tag}.csv", index=False)
    logger.info(f"Saved {len(rows)} rows for judge {args.judge}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()

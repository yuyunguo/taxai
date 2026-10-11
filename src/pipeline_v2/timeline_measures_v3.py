"""Descriptive timeline measures promised in plan/analysis-plan-v3.md Section 2 (Amendment A1).

Per case and arm:
  * input contrast: lab values beyond first/last per test plus vital points
    beyond min/max per variable (what the chart adds beyond the summary);
  * time references and sequence terms per explanation;
  * the generator's directional verdict and whether it is right:
      label wrong and HIGH -> "too high" is right
      label wrong and LOW  -> "too low" is right
      label correct        -> "about right" is right

Usage:
    python -m src.pipeline_v2.timeline_measures_v3 --set main
"""

import argparse
import json
import logging
import pickle
import re

import pandas as pd

from .config import DATA_DIR
from .pilot_gates_v3 import input_contrast

logger = logging.getLogger(__name__)
TIME = re.compile(r"\b(icu )?hours? ?[+-]?\d+|\bat (icu )?hour|\bover (the )?(\d+|first|next|past)\s*(h|hours)\b"
                  r"|\bwithin \d+ ?h", re.I)
SEQ = re.compile(r"\b(then|subsequently|after|before|followed by|initially|later|by (icu )?hour)\b", re.I)
VERDICT = re.compile(r"appears (too high|about right|too low)", re.I)


def right_verdict(level: str, mortality: int) -> str:
    correct = (level == "HIGH") == (mortality == 1)
    if correct:
        return "about right"
    return "too high" if level == "HIGH" else "too low"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["pilot", "main"], required=True)
    args = ap.parse_args()
    od = DATA_DIR / "v3" / args.set
    cases = json.load(open(od / "sample_meta.json"))["cases"]
    expl = json.load(open(od / "explanations.json"))
    with open(DATA_DIR / "trajectories_test_v3.pkl", "rb") as f:
        trajs = pickle.load(f)
    rows = []
    for c in cases:
        h = c["hadm_id"]
        contrast = input_contrast(trajs[h])
        target = right_verdict(c["level"], c["mortality"])
        for m in ("static_llm", "taxai"):
            text = expl[m][str(h)]
            v = VERDICT.search(text)
            verdict = v.group(1).lower() if v else "missing"
            rows.append({"hadm_id": h, "method": m, "input_contrast": contrast,
                         "time_refs": len(TIME.findall(text)), "sequence_terms": len(SEQ.findall(text)),
                         "verdict": verdict, "verdict_right": verdict == target, "right_verdict": target})
    df = pd.DataFrame(rows)
    df.to_csv(od / "timeline_measures_v3.csv", index=False)
    summary = {
        "input_contrast_median_iqr": df.drop_duplicates("hadm_id")["input_contrast"].quantile([.5, .25, .75]).tolist(),
        "by_method": {m: {"time_refs_mean": round(float(d["time_refs"].mean()), 3),
                          "any_time_ref": round(float((d["time_refs"] > 0).mean()), 3),
                          "sequence_terms_mean": round(float(d["sequence_terms"].mean()), 3),
                          "verdicts": d["verdict"].value_counts().to_dict(),
                          "verdict_accuracy": round(float(d["verdict_right"].mean()), 4)}
                      for m, d in df.groupby("method")},
    }
    json.dump(summary, open(od / "timeline_measures_v3.json", "w"), indent=2)
    logger.info(json.dumps(summary, indent=1))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()

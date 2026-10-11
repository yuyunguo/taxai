"""Generate v3 explanations for the main sample (the v2 500 cases) or the pilot.

Pilot: 20 HIGH + 20 LOW test cases disjoint from the main sample (seed 7).
Static LLM receives the aggregate summary; TAXAI receives the summary plus
the full chronological record; both use the identical EXPLAIN_V3 instruction.

Usage:
    python -m src.pipeline_v2.explain_v3 --set pilot
    python -m src.pipeline_v2.explain_v3 --set main
"""

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np

from .config import DATA_DIR, LLM
from .explain import shap_explanations
from .llm import CachedLLM
from .prompts import prediction_line
from .prompts_v3 import EXPLAIN_V3, render_chart, render_summary

logger = logging.getLogger(__name__)
PILOT_PER_LABEL, PILOT_SEED = 20, 7


def out_dir(which: str) -> Path:
    d = DATA_DIR / "v3" / which
    d.mkdir(parents=True, exist_ok=True)
    return d


def select_cases(which: str, trajs: Dict[int, dict], preds: Dict[int, float], thr: float) -> List[dict]:
    main = json.load(open(DATA_DIR / "explanations" / "sample_meta.json"))["cases"]
    if which == "main":
        return main
    used = {c["hadm_id"] for c in main}
    rng = np.random.default_rng(PILOT_SEED)
    ids = np.array(sorted(h for h in trajs if h not in used))
    p = np.array([preds[h] for h in ids])
    chosen: List[int] = []
    for mask in (p >= thr, p < thr):
        chosen += rng.choice(ids[mask], PILOT_PER_LABEL, replace=False).tolist()
    return [{"hadm_id": int(h), "pred_proba": round(preds[h], 4), "level": "HIGH" if preds[h] >= thr else "LOW",
             "mortality": trajs[h]["mortality"]} for h in chosen]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["pilot", "main"], required=True)
    args = ap.parse_args()
    with open(DATA_DIR / "trajectories_test_v3.pkl", "rb") as f:
        trajs = pickle.load(f)
    with open(DATA_DIR / "predictions.pkl", "rb") as f:
        preds = pickle.load(f)
    art = joblib.load(DATA_DIR / "models" / "xgboost_v2.pkl")
    thr = art["threshold"]
    cases = select_cases(args.set, trajs, preds, thr)
    od = out_dir(args.set)
    json.dump({"threshold": thr, "cases": cases}, open(od / "sample_meta.json", "w"), indent=2)
    hids = [c["hadm_id"] for c in cases]

    expl: Dict[str, Dict[int, str]] = {"empty": {h: "" for h in hids}}
    rng = np.random.default_rng(PILOT_SEED + 1 if args.set == "pilot" else 43)
    donors = rng.choice([h for h in trajs if h not in set(hids)], size=len(hids), replace=False)
    expl["random"] = {h: render_summary(trajs[int(d)]) for h, d in zip(hids, donors)}
    expl["shap"] = shap_explanations(hids, art)
    llm = CachedLLM(od / "generation_cache.jsonl")
    expl["static_llm"], expl["taxai"] = {}, {}
    for i, c in enumerate(cases):
        h, t = c["hadm_id"], trajs[c["hadm_id"]]
        pl = prediction_line(c["pred_proba"], c["level"], thr)
        summary = f"TRAJECTORY SUMMARY:\n{render_summary(t)}"
        full = f"{summary}\n\n{render_chart(t)}"
        expl["static_llm"][h] = llm.ask(LLM.generator, "explain_v3", "static_llm", h,
                                        EXPLAIN_V3.format(prediction=pl, data=summary, level=c["level"]))
        expl["taxai"][h] = llm.ask(LLM.generator, "explain_v3", "taxai", h,
                                   EXPLAIN_V3.format(prediction=pl, data=full, level=c["level"]))
        if (i + 1) % 50 == 0:
            logger.info(f"  generated {i + 1}/{len(cases)}")
    json.dump({m: {str(k): v for k, v in d.items()} for m, d in expl.items()}, open(od / "explanations.json", "w"))
    logger.info(f"v3 explanations saved for {len(cases)} cases ({args.set})")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()

"""Select the evaluation sample and generate all five explanation methods.

Sample: from the temporal test split, n_per_label admissions predicted HIGH
and n_per_label predicted LOW (threshold from validation), seed-fixed.

Methods: empty, random (another test patient's summary), SHAP (top-10
TreeExplainer attributions with feature values), static_llm (summary only),
taxai (summary + chronological timeline). The two LLM methods share the same
instruction text.

Usage:
    python -m src.pipeline_v2.explain
"""

import json
import logging
import pickle
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
import shap

from .config import DATA_DIR, LLM, STUDY
from .llm import CachedLLM
from .prompts import (EXPLAIN, PATIENT, REGULATOR, prediction_line, render_summary,
                      render_timeline)

logger = logging.getLogger(__name__)


def select_sample(preds: Dict[int, float], trajs: Dict[int, dict], thr: float) -> List[dict]:
    rng = np.random.default_rng(STUDY.seed)
    ids = np.array(sorted(trajs))
    p = np.array([preds[i] for i in ids])
    chosen: List[int] = []
    for mask in (p >= thr, p < thr):
        pool = ids[mask]
        chosen += rng.choice(pool, size=min(STUDY.n_per_label, len(pool)), replace=False).tolist()
    return [{"hadm_id": int(h), "pred_proba": round(preds[h], 4),
             "level": "HIGH" if preds[h] >= thr else "LOW",
             "mortality": trajs[h]["mortality"], "n_events": len(trajs[h]["events"])}
            for h in chosen]


def shap_explanations(hids: List[int], art: dict) -> Dict[int, str]:
    feats = pd.read_parquet(DATA_DIR / "features.parquet").set_index("hadm_id").loc[hids]
    cols = art["feature_cols"]
    raw = feats[cols].values.astype(np.float32)
    X = art["imputer"].transform(raw)
    explainer = shap.TreeExplainer(art["model"])
    sv = explainer.shap_values(X)
    base = float(1 / (1 + np.exp(-explainer.expected_value)))
    out: Dict[int, str] = {}
    for i, hid in enumerate(hids):
        top = np.argsort(-np.abs(sv[i]))[:10]
        lines = []
        for j in top:
            val = raw[i, j]
            shown = "missing" if np.isnan(val) else f"{val:.3g}"
            direction = "increases" if sv[i, j] > 0 else "decreases"
            lines.append(f"  {cols[j]} = {shown} {direction} risk (SHAP {sv[i, j]:+.3f} log-odds)")
        out[hid] = f"Model baseline risk: {base:.1%}\nTop contributing features:\n" + "\n".join(lines)
    return out


def main() -> None:
    with open(DATA_DIR / "trajectories_test.pkl", "rb") as f:
        trajs = pickle.load(f)
    with open(DATA_DIR / "predictions.pkl", "rb") as f:
        preds = pickle.load(f)
    art = joblib.load(DATA_DIR / "models" / "xgboost_v2.pkl")
    thr = art["threshold"]

    sample = select_sample(preds, trajs, thr)
    hids = [s["hadm_id"] for s in sample]
    out_dir = DATA_DIR / "explanations"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "sample_meta.json", "w") as f:
        json.dump({"threshold": thr, "cases": sample}, f, indent=2)
    logger.info(f"Sample: {len(sample)} cases; deaths {sum(s['mortality'] for s in sample)}")

    expl: Dict[str, Dict[int, str]] = {"empty": {h: "" for h in hids}}
    rng = np.random.default_rng(STUDY.seed + 1)
    others = np.array([h for h in trajs if h not in set(hids)])
    donors = rng.choice(others, size=len(hids), replace=False)
    expl["random"] = {h: render_summary(trajs[int(d)]) for h, d in zip(hids, donors)}
    expl["shap"] = shap_explanations(hids, art)

    llm = CachedLLM(out_dir / "generation_cache.jsonl")
    expl["static_llm"], expl["taxai"] = {}, {}
    multi: Dict[int, Dict[str, str]] = {}
    for i, s in enumerate(sample):
        h, t = s["hadm_id"], trajs[s["hadm_id"]]
        pl = prediction_line(s["pred_proba"], s["level"], thr)
        summary = render_summary(t)
        full = f"TRAJECTORY SUMMARY:\n{summary}\n\nCHRONOLOGICAL EVENTS:\n{render_timeline(t)}"
        expl["static_llm"][h] = llm.ask(LLM.generator, "explain", "static_llm", h, EXPLAIN.format(
            prediction=pl, data=f"TRAJECTORY SUMMARY:\n{summary}", level=s["level"]))
        expl["taxai"][h] = llm.ask(LLM.generator, "explain", "taxai", h, EXPLAIN.format(
            prediction=pl, data=full, level=s["level"]))
        if i < STUDY.n_multilevel:
            multi[h] = {
                "clinician": expl["taxai"][h],
                "patient": llm.ask(LLM.generator, "explain", "taxai_patient", h,
                                   PATIENT.format(prediction=pl, data=full)),
                "regulator": llm.ask(LLM.generator, "explain", "taxai_regulator", h,
                                     REGULATOR.format(prediction=pl, data=full)),
            }
        if (i + 1) % 50 == 0:
            logger.info(f"  generated {i + 1}/{len(sample)}")

    with open(out_dir / "explanations.json", "w") as f:
        json.dump({m: {str(k): v for k, v in d.items()} for m, d in expl.items()}, f)
    with open(out_dir / "taxai_multilevel.json", "w") as f:
        json.dump({str(k): v for k, v in multi.items()}, f, indent=2)
    logger.info("Explanations saved.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()

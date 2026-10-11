"""Limited Gate 2 test on invasive ventilation (plan/gate2-imv.md; exploratory).

Trains model A (aggregates) and B (+dynamics) for 24-h invasive ventilation,
labels test patients HIGH/LOW by A at the validation-F1 threshold, draws an
outcome-enriched sample per stratum, generates Static LLM and TAXAI
explanations with a ventilation-specific prompt, and scores the verdicts.

Usage (on the database host):
    PGPASSWORD=... python -m src.pipeline_v2.gate2_imv
"""

import json
import logging
import pickle
import re
from typing import Dict

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import precision_recall_curve, roc_auc_score

from .build import load_events
from .config import DATA_DIR, LLM
from .gate1_outcomes import outcome_labels
from .info_ceiling import dynamics_features, fit
from .llm import CachedLLM
from .prompts_v3 import EXPLAIN_V3, render_chart, render_summary

logger = logging.getLogger(__name__)
SCORE = {"too high": -1, "about right": 0, "too low": 1}
VERDICT = re.compile(r"appears (too high|about right|too low)", re.I)
PER_STRATUM_EVENTS = 60

EXPLAIN_IMV = (EXPLAIN_V3
               .replace("most relevant to mortality risk", "most relevant to the need for invasive mechanical ventilation in the next 24 hours"))


def prediction_line_imv(p: float, level: str, thr: float) -> str:
    return (f"AI PREDICTION: {p:.1%} risk that this patient will need invasive mechanical ventilation "
            f"in the next 24 hours (labeled {level} risk; the system's HIGH-risk threshold is {thr:.0%}).")


def auroc_ci(y: np.ndarray, s: np.ndarray, seed: int = 42) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    n, b = len(y), []
    for _ in range(1000):
        i = rng.integers(0, n, n)
        if y[i].min() != y[i].max():
            b.append(roc_auc_score(y[i], s[i]))
    return {"auroc": round(float(roc_auc_score(y, s)), 4), "ci95": np.percentile(b, [2.5, 97.5]).round(4).tolist()}


def paired_diff(y: np.ndarray, a: np.ndarray, b: np.ndarray, seed: int = 42) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    n, d = len(y), []
    for _ in range(1000):
        i = rng.integers(0, n, n)
        if y[i].min() != y[i].max():
            d.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    return {"diff": round(float(roc_auc_score(y, a) - roc_auc_score(y, b)), 4),
            "ci95": np.percentile(d, [2.5, 97.5]).round(4).tolist()}


def main() -> None:
    cohort = pd.read_parquet(DATA_DIR / "cohort.parquet")
    feat = pd.read_parquet(DATA_DIR / "features.parquet").set_index("hadm_id")
    dyn = dynamics_features(load_events(cohort))
    full = feat.join(dyn)
    base = [c for c in feat.columns if c not in ("mortality", "split")]
    lab = outcome_labels(cohort)["invasive_ventilation"]
    d = full.loc[lab["hadm_id"]].copy()
    d["y"] = lab["event"].values
    parts = {s: d[d["split"] == s] for s in ("train", "val", "test")}
    y = {s: parts[s]["y"].values for s in parts}
    prob = {}
    for key, cols in (("A", base), ("B", base + list(dyn.columns))):
        X = {s: parts[s][cols].values.astype(np.float32) for s in parts}
        m = fit(X, y)
        prob[key] = {s: m.predict_proba(X[s])[:, 1] for s in ("val", "test")}
    prec, rec, thr_arr = precision_recall_curve(y["val"], prob["A"]["val"])
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-9, None)
    thr = float(thr_arr[int(np.argmax(f1))])
    test = parts["test"][["y"]].assign(pA=prob["A"]["test"], pB=prob["B"]["test"])
    test["level"] = np.where(test["pA"] >= thr, "HIGH", "LOW")
    logger.info(f"threshold {thr:.4f}; test HIGH {int((test.level == 'HIGH').sum())} "
                f"(events {int(test[test.level == 'HIGH'].y.sum())}), LOW events {int(test[test.level == 'LOW'].y.sum())}")

    rng = np.random.default_rng(11)
    picks = []
    for level in ("HIGH", "LOW"):
        s = test[test["level"] == level]
        ev, ne = s[s["y"] == 1], s[s["y"] == 0]
        k = min(len(ev), PER_STRATUM_EVENTS, len(ne))
        picks += list(rng.choice(ev.index, k, replace=False)) + list(rng.choice(ne.index, k, replace=False))
    sample = test.loc[picks].copy()
    with open(DATA_DIR / "trajectories_test_v3.pkl", "rb") as f:
        trajs = pickle.load(f)
    out_dir = DATA_DIR / "v3" / "gate2_imv"
    out_dir.mkdir(parents=True, exist_ok=True)
    llm = CachedLLM(out_dir / "generation_cache.jsonl")
    rows = []
    for i, (hid, r) in enumerate(sample.iterrows()):
        t = trajs[int(hid)]
        pl = prediction_line_imv(float(r["pA"]), r["level"], thr)
        summary = f"TRAJECTORY SUMMARY:\n{render_summary(t)}"
        for method, data in (("static_llm", summary), ("taxai", f"{summary}\n\n{render_chart(t)}")):
            text = llm.ask(LLM.generator, "explain_imv", method, int(hid),
                           EXPLAIN_IMV.format(prediction=pl, data=data, level=r["level"]))
            v = VERDICT.search(text)
            rows.append({"hadm_id": int(hid), "method": method, "level": r["level"], "y": int(r["y"]),
                         "pA": float(r["pA"]), "pB": float(r["pB"]), "verdict": v.group(1).lower() if v else "missing"})
        if (i + 1) % 40 == 0:
            logger.info(f"  {i + 1}/{len(sample)}")
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "gate2_imv_cases.csv", index=False)

    res: Dict[str, object] = {"threshold": round(thr, 4), "n_cases": int(len(sample))}
    for level in ("HIGH", "LOW"):
        s = df[df["level"] == level]
        if s["y"].nunique() < 2:
            continue
        w = s.pivot(index="hadm_id", columns="method", values="verdict").map(lambda v: SCORE.get(v, 0))
        yy = s.drop_duplicates("hadm_id").set_index("hadm_id").loc[w.index]
        r: Dict[str, object] = {"n": int(len(w)), "events": int(yy["y"].sum()),
                                "model_A": auroc_ci(yy["y"].values, yy["pA"].values),
                                "model_B": auroc_ci(yy["y"].values, yy["pB"].values)}
        for m in ("static_llm", "taxai"):
            r[m] = auroc_ci(yy["y"].values, w[m].values)
            r[f"{m}_verdicts"] = s[s["method"] == m]["verdict"].value_counts().to_dict()
        r["taxai_minus_static"] = paired_diff(yy["y"].values, w["taxai"].values, w["static_llm"].values)
        res[level] = r
    s = df[df["method"] == "taxai"].copy()
    s["score"] = s["verdict"].map(lambda v: SCORE.get(v, 0))
    lg = np.log(np.clip(s["pA"], 1e-4, 1 - 1e-4) / (1 - np.clip(s["pA"], 1e-4, 1 - 1e-4)))
    fitlr = sm.Logit(s["y"].values, sm.add_constant(np.column_stack([lg, s["score"].values]))).fit(disp=0)
    res["incremental_taxai_verdict"] = {"coef": round(float(fitlr.params[2]), 4), "p": float(fitlr.pvalues[2])}
    main_stratum = max(("HIGH", "LOW"), key=lambda l: res.get(l, {}).get("events", -1))
    m = res[main_stratum]
    res["decision"] = {"main_stratum": main_stratum,
                       "gate2_open_taxai": bool(m["taxai"]["auroc"] >= 0.60 and m["taxai"]["ci95"][0] > 0.5),
                       "timeline_helps_explainer": bool(m["taxai_minus_static"]["ci95"][0] > 0)}
    json.dump(res, open(out_dir / "gate2_imv.json", "w"), indent=2)
    logger.info(json.dumps(res, indent=1))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()

"""Post hoc Gate 1 and outcome analyses after the simulated review (plan/posthoc-v4-tierA.md, P5-P7).

P5: source of the ventilation Gate 1 signal (feature-family ablation, capacity control, strata).
P6: ventilation model A performance and calibration.
P7: ventilation event characterization, code status, competing events, and the DNI/comfort-care
    sensitivity analysis (Gate 1 here; Gate 2 via posthoc_v4_gate2 --exclude-ids).

Usage (on the database host):
    PGPASSWORD=... python -m src.pipeline_v2.posthoc_v4_gate1
"""

import json
import logging
from typing import Dict, List

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import brier_score_loss, roc_auc_score
from sqlalchemy import bindparam, create_engine, text

from .build import load_events
from .config import DATA_DIR, DB
from .gate1_outcomes import outcome_labels
from .gate2_v4 import OUT, _threshold
from .info_ceiling import dynamics_features, fit, paired_boot

logger = logging.getLogger(__name__)
PROCESS = ("time_since_last", "n_last12")
LIMITING = ("DNR / DNI", "DNI (do not intubate)", "Comfort measures only")


def auroc_ci(y: np.ndarray, p: np.ndarray, seed: int = 42) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    b = [roc_auc_score(y[i], p[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(1000)) if y[i].min() != y[i].max()]
    return {"est": round(float(roc_auc_score(y, p)), 4), "ci95": np.percentile(b, [2.5, 97.5]).round(4).tolist()}


def calibration(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    slope = sm.Logit(y, sm.add_constant(lp)).fit(disp=0).params[1]
    intercept = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=lp).fit().params[0]
    return {"intercept": round(float(intercept), 3), "slope": round(float(slope), 3),
            "observed": round(float(y.mean()), 4), "mean_predicted": round(float(p.mean()), 4),
            "brier": round(float(brier_score_loss(y, p)), 4)}


def fit_predict(d: pd.DataFrame, cols: List[str]) -> Dict[str, np.ndarray]:
    parts = {s: d[d["split"] == s] for s in ("train", "val", "test")}
    X = {s: parts[s][cols].values.astype(np.float32) for s in parts}
    y = {s: parts[s]["y"].values for s in parts}
    m = fit(X, y)
    return {s: m.predict_proba(X[s])[:, 1] for s in ("val", "test")}


def outcome_details(cohort: pd.DataFrame, ids: pd.Series) -> pd.DataFrame:
    """Per eligible stay: event time, NIV/HFNC exposure, limiting code status, death/discharge within 24 h."""
    engine = create_engine(DB.uri)
    sql = """
      SELECT c.stay_id,
        MIN(CASE WHEN v.ventilation_status='InvasiveVent' AND v.starttime > c.pred_time
                  AND v.starttime <= c.pred_time + interval '24 hours'
                 THEN EXTRACT(EPOCH FROM v.starttime - c.pred_time)/3600 END) AS hours_to_imv,
        MAX(CASE WHEN v.ventilation_status IN ('NonInvasiveVent','HFNC')
                  AND v.starttime <= c.pred_time + interval '24 hours'
                  AND v.endtime >= c.pred_time - interval '24 hours' THEN 1 ELSE 0 END) AS niv_hfnc
      FROM cohort c LEFT JOIN mimic_derived.ventilation v ON v.stay_id = c.stay_id GROUP BY c.stay_id"""
    sql_code = """
      SELECT c.stay_id, MAX(CASE WHEN ce.value IN :lim THEN 1 ELSE 0 END) AS limiting_code
      FROM cohort c JOIN mimic_icu.chartevents ce ON ce.stay_id = c.stay_id
      WHERE ce.itemid IN (223758, 228687) AND ce.charttime <= c.pred_time GROUP BY c.stay_id"""
    sql_comp = """
      SELECT c.stay_id,
        MAX(CASE WHEN a.deathtime > c.pred_time AND a.deathtime <= c.pred_time + interval '24 hours' THEN 1 ELSE 0 END) AS death_24h,
        MAX(CASE WHEN i.outtime <= c.pred_time + interval '24 hours' THEN 1 ELSE 0 END) AS icu_out_24h
      FROM cohort c JOIN mimic_icu.icustays i ON i.stay_id = c.stay_id
      JOIN mimic_core.admissions a ON a.hadm_id = i.hadm_id GROUP BY c.stay_id"""
    sub = cohort[cohort["hadm_id"].isin(ids)]
    with engine.connect() as conn:
        conn.execute(text("CREATE TEMP TABLE cohort (stay_id bigint, pred_time timestamp)"))
        conn.execute(text("INSERT INTO cohort VALUES (:s, :t)"),
                     [{"s": int(s), "t": t.to_pydatetime()} for s, t in zip(sub["stay_id"], sub["pred_time"])])
        a = pd.read_sql(text(sql), conn)
        b = pd.read_sql(text(sql_code).bindparams(bindparam("lim", expanding=True)), conn, params={"lim": list(LIMITING)})
        c = pd.read_sql(text(sql_comp), conn)
    out = sub[["hadm_id", "stay_id"]].merge(a, on="stay_id", how="left").merge(b, on="stay_id", how="left")
    out = out.merge(c, on="stay_id", how="left").fillna({"niv_hfnc": 0, "limiting_code": 0})
    return out


def main() -> None:
    cohort = pd.read_parquet(DATA_DIR / "cohort.parquet")
    feat = pd.read_parquet(DATA_DIR / "features.parquet").set_index("hadm_id")
    dyn = dynamics_features(load_events(cohort))
    base = [c for c in feat.columns if c not in ("mortality", "split")]
    proc = [c for c in dyn.columns if any(c.endswith(f"_dyn_{p}") for p in PROCESS)]
    value = [c for c in dyn.columns if c not in proc]
    lab = outcome_labels(cohort)["invasive_ventilation"]
    d = feat.join(dyn).loc[lab["hadm_id"]].copy()
    d["y"] = lab["event"].values
    rng = np.random.default_rng(42)
    perm = d[list(dyn.columns)].copy()
    for s in ("train", "val", "test"):
        idx = perm.index[d["split"] == s]
        perm.loc[idx] = perm.loc[idx].values[rng.permutation(len(idx))]
    shuffled_cols = [f"perm__{c}" for c in dyn.columns]
    d[shuffled_cols] = perm.values
    y_test = d[d["split"] == "test"]["y"].values
    preds = {"A": fit_predict(d, base), "A+value": fit_predict(d, base + value),
             "A+process": fit_predict(d, base + proc), "B": fit_predict(d, base + list(dyn.columns)),
             "A+permuted": fit_predict(d, base + shuffled_cols)}
    res: Dict[str, object] = {"n_value_features": len(value), "n_process_features": len(proc)}
    res["P5_ablation_test"] = {k: paired_boot(y_test, preds["A"]["test"], v["test"], roc_auc_score)
                               for k, v in preds.items() if k != "A"}
    thr = _threshold(d[d["split"] == "val"]["y"].values, preds["A"]["val"])
    lvl = np.where(preds["A"]["test"] >= thr, "HIGH", "LOW")
    res["P5_within_strata_test"] = {s: {"n": int((lvl == s).sum()), "events": int(y_test[lvl == s].sum()),
                                        **paired_boot(y_test[lvl == s], preds["A"]["test"][lvl == s],
                                                      preds["B"]["test"][lvl == s], roc_auc_score)}
                                    for s in ("LOW", "HIGH")}
    y_val = d[d["split"] == "val"]["y"].values
    res["P6_model_A"] = {"threshold": round(thr, 4), "test_auroc": auroc_ci(y_test, preds["A"]["test"]),
                         "val_auroc": round(float(roc_auc_score(y_val, preds["A"]["val"])), 4),
                         "test_calibration": calibration(y_test, preds["A"]["test"]),
                         "test_LOW_calibration": calibration(y_test[lvl == "LOW"], preds["A"]["test"][lvl == "LOW"])}
    det = outcome_details(cohort, lab["hadm_id"]).merge(lab[["hadm_id", "event", "split"]], on="hadm_id")
    ev, ne = det[det["event"] == 1], det[det["event"] == 0]
    q = ev["hours_to_imv"].quantile([0.25, 0.5, 0.75]).round(1).tolist()
    res["P7_events"] = {
        "n_events": int(len(ev)), "hours_to_imv_iqr": q,
        "within_2h": round(float((ev["hours_to_imv"] <= 2).mean()), 4),
        "niv_hfnc_events": round(float(ev["niv_hfnc"].mean()), 4), "niv_hfnc_nonevents": round(float(ne["niv_hfnc"].mean()), 4),
        "limiting_code_events": int(ev["limiting_code"].sum()), "limiting_code_nonevents": int(ne["limiting_code"].sum()),
        "nonevents_death_24h": int(ne["death_24h"].sum()), "nonevents_icu_out_24h": int(ne["icu_out_24h"].sum()),
        "n_nonevents": int(len(ne))}
    excl = set(det.loc[det["limiting_code"] == 1, "hadm_id"].astype(int))
    keep = ~d.index.isin(excl)
    dk = d[keep]
    yk = dk[dk["split"] == "test"]["y"].values
    pk = {k: fit_predict(dk, cols) for k, cols in (("A", base), ("B", base + list(dyn.columns)))}
    res["P7d_gate1_excluding_limiting_code"] = {"n_excluded": len(excl),
                                                **paired_boot(yk, pk["A"]["test"], pk["B"]["test"], roc_auc_score)}
    with open(OUT / "posthoc_limiting_code_ids.txt", "w") as f:
        f.write("\n".join(str(i) for i in sorted(excl)))
    json.dump(res, open(OUT / "posthoc_v4_gate1.json", "w"), indent=2)
    logger.info(json.dumps(res, indent=1))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()

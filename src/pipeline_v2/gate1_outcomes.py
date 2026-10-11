"""Limited Gate 1 test on timing-driven outcomes (plan/gate1-other-outcomes.md).

Outcomes in the 24 h after the prediction time: norepinephrine initiation and
invasive ventilation start, each excluding patients already exposed before the
prediction time. Compares aggregates (A) vs aggregates + dynamics (B).

Usage (on the database host):
    PGPASSWORD=... python -m src.pipeline_v2.gate1_outcomes
"""

import json
import logging
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sqlalchemy import create_engine, text

from .build import load_events
from .config import DATA_DIR, DB
from .info_ceiling import dynamics_features, fit, paired_boot

logger = logging.getLogger(__name__)
HORIZON_H = 24

OUTCOME_SQL = {
    "norepinephrine": """
        SELECT c.stay_id,
               MAX(CASE WHEN n.starttime <= c.pred_time THEN 1 ELSE 0 END) AS prior,
               MAX(CASE WHEN n.starttime > c.pred_time
                         AND n.starttime <= c.pred_time + make_interval(hours => :h) THEN 1 ELSE 0 END) AS event
        FROM cohort c JOIN mimic_derived.norepinephrine n ON n.stay_id = c.stay_id
        GROUP BY c.stay_id""",
    "invasive_ventilation": """
        SELECT c.stay_id,
               MAX(CASE WHEN v.starttime <= c.pred_time THEN 1 ELSE 0 END) AS prior,
               MAX(CASE WHEN v.ventilation_status = 'InvasiveVent' AND v.starttime > c.pred_time
                         AND v.starttime <= c.pred_time + make_interval(hours => :h) THEN 1 ELSE 0 END) AS event
        FROM cohort c JOIN mimic_derived.ventilation v ON v.stay_id = c.stay_id
        WHERE v.ventilation_status IN ('InvasiveVent', 'Tracheostomy')
        GROUP BY c.stay_id""",
}


def outcome_labels(cohort: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    engine = create_engine(DB.uri)
    out = {}
    with engine.connect() as conn:
        conn.execute(text("CREATE TEMP TABLE cohort (stay_id bigint, pred_time timestamp)"))
        conn.execute(text("INSERT INTO cohort VALUES (:s, :t)"),
                     [{"s": int(s), "t": t.to_pydatetime()} for s, t in zip(cohort["stay_id"], cohort["pred_time"])])
        for name, sql in OUTCOME_SQL.items():
            lab = pd.read_sql(text(sql), conn, params={"h": HORIZON_H})
            df = cohort[["hadm_id", "stay_id", "split"]].merge(lab, on="stay_id", how="left").fillna({"prior": 0, "event": 0})
            df = df[df["prior"] == 0]
            out[name] = df[["hadm_id", "split", "event"]].astype({"event": int})
            logger.info(f"{name}: eligible {len(df):,}; events {int(df['event'].sum()):,} ({df['event'].mean():.1%})")
    return out


def main() -> None:
    cohort = pd.read_parquet(DATA_DIR / "cohort.parquet")
    feat = pd.read_parquet(DATA_DIR / "features.parquet").set_index("hadm_id")
    dyn = dynamics_features(load_events(cohort))
    full = feat.join(dyn)
    base_cols = [c for c in feat.columns if c not in ("mortality", "split")]
    dyn_cols = list(dyn.columns)
    res: Dict[str, object] = {}
    for name, lab in outcome_labels(cohort).items():
        d = full.loc[lab["hadm_id"]].copy()
        d["y"] = lab["event"].values
        parts = {s: d[d["split"] == s] for s in ("train", "val", "test")}
        y = {s: parts[s]["y"].values for s in parts}
        p = {}
        for key, cols in (("A", base_cols), ("B", base_cols + dyn_cols)):
            X = {s: parts[s][cols].values.astype(np.float32) for s in parts}
            p[key] = fit(X, y).predict_proba(X["test"])[:, 1]
        a = paired_boot(y["test"], p["A"], p["B"], roc_auc_score)
        res[name] = {"n_test": int(len(y["test"])), "events_test": int(y["test"].sum()),
                     "prevalence_test": round(float(y["test"].mean()), 4),
                     "auroc": a, "auprc": paired_boot(y["test"], p["A"], p["B"], average_precision_score),
                     "gate1_open": bool(a["diff"] >= 0.010 and a["ci95"][0] > 0)}
        logger.info(f"{name}: {res[name]}")
    json.dump(res, open(DATA_DIR / "gate1_other_outcomes.json", "w"), indent=2)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()

"""External validation: apply the MIMIC-IV base model, unchanged, to eICU.

No refitting or recalibration: the MIMIC-trained model, imputer, and
validation-chosen threshold are applied as-is. Reports discrimination,
calibration, and label accuracy overall and across hospitals, and writes
predictions.pkl so the explanation study can be repeated on eICU.

Usage:
    TAXAI_V2_DATA=.../data_eicu python -m src.pipeline_v2.eicu_apply \
        --model .../data_v2/models/xgboost_v2.pkl
"""

import argparse
import json
import logging
import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .config import DATA_DIR
from .model import _metrics

logger = logging.getLogger(__name__)
MIN_HOSPITAL_N, MIN_HOSPITAL_DEATHS = 100, 10


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    args = ap.parse_args()

    art = joblib.load(args.model)
    cols, thr = art["feature_cols"], art["threshold"]
    feat = pd.read_parquet(DATA_DIR / "features.parquet")
    missing = [c for c in cols if c not in feat.columns]
    for c in missing:
        feat[c] = np.nan
    all_nan = [c for c in cols if feat[c].isna().all()]
    logger.info(f"{len(missing)} model features absent in eICU; {len(all_nan)} entirely missing: {all_nan[:12]}")

    X = art["imputer"].transform(feat[cols].values.astype(np.float32))
    p = art["model"].predict_proba(X)[:, 1]
    y = feat["mortality"].values
    results = {"n": int(len(y)), "threshold": thr, "overall": _metrics(y, p, thr),
               "features_entirely_missing": all_nan}

    cohort = pd.read_parquet(DATA_DIR / "cohort.parquet")[["hadm_id", "hospitalid"]]
    df = feat[["hadm_id"]].assign(p=p, y=y).merge(cohort, on="hadm_id")
    per_h = []
    for hid, d in df.groupby("hospitalid"):
        if len(d) >= MIN_HOSPITAL_N and d["y"].sum() >= MIN_HOSPITAL_DEATHS:
            per_h.append({"hospitalid": int(hid), "n": len(d), "deaths": int(d["y"].sum()),
                          "auroc": float(roc_auc_score(d["y"], d["p"])),
                          "obs_exp_ratio": float(d["y"].mean() / d["p"].mean())})
    ph = pd.DataFrame(per_h)
    if len(ph):
        results["per_hospital"] = {
            "n_hospitals": len(ph),
            "auroc_median_iqr": [round(float(x), 4) for x in ph["auroc"].quantile([0.5, 0.25, 0.75])],
            "obs_exp_median_iqr": [round(float(x), 4) for x in ph["obs_exp_ratio"].quantile([0.5, 0.25, 0.75])],
        }
        ph.to_csv(DATA_DIR / "external_per_hospital.csv", index=False)
    logger.info(json.dumps(results, indent=1))

    with open(DATA_DIR / "predictions.pkl", "wb") as f:
        pickle.dump({int(h): float(v) for h, v in zip(feat["hadm_id"], p)}, f)
    (DATA_DIR / "models").mkdir(exist_ok=True)
    joblib.dump(art, DATA_DIR / "models" / "xgboost_v2.pkl")   # same artifact, for explain.py
    with open(DATA_DIR / "external_validation.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()

"""Bootstrap CIs, subgroup performance, and cohort demographics for the base model.

Closes TRIPOD+AI items 20b/20c/23a: AUROC, AUPRC, and Brier with 95%
bootstrap CIs for MIMIC-IV validation/test and eICU; AUROC by sex and age
group; demographics for every cohort.

Usage:
    python -m src.pipeline_v2.model_uncertainty --mimic DIR --eicu DIR --out FILE
"""

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

logger = logging.getLogger(__name__)
N_BOOT, SEED = 1000, 42
AGE_BINS = [(18, 64, "18-64"), (65, 79, "65-79"), (80, 200, "80+")]


def boot(y: np.ndarray, p: np.ndarray, fn) -> Dict[str, object]:
    rng = np.random.default_rng(SEED)
    n, vals = len(y), []
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        if y[i].min() != y[i].max():
            vals.append(fn(y[i], p[i]))
    return {"est": round(float(fn(y, p)), 4), "ci95": [round(float(v), 4) for v in np.percentile(vals, [2.5, 97.5])]}


def evaluate(df: pd.DataFrame) -> Dict[str, object]:
    y, p = df["y"].values, df["p"].values
    out: Dict[str, object] = {"n": int(len(df)), "deaths": int(y.sum()),
                              "auroc": boot(y, p, roc_auc_score),
                              "auprc": boot(y, p, average_precision_score),
                              "brier": boot(y, p, brier_score_loss)}
    sub = {}
    for sex, label in (("F", "female"), ("M", "male")):
        d = df[df["gender"] == sex]
        if d["y"].nunique() > 1:
            sub[label] = {"n": int(len(d)), "deaths": int(d["y"].sum()), "auroc": boot(d["y"].values, d["p"].values, roc_auc_score)}
    for lo, hi, label in AGE_BINS:
        d = df[(df["age"] >= lo) & (df["age"] <= hi)]
        if d["y"].nunique() > 1:
            sub[f"age_{label}"] = {"n": int(len(d)), "deaths": int(d["y"].sum()),
                                   "auroc": boot(d["y"].values, d["p"].values, roc_auc_score)}
    out["subgroups"] = sub
    return out


def demographics(c: pd.DataFrame) -> Dict[str, object]:
    q = c["age"].quantile([0.25, 0.5, 0.75])
    return {"n": int(len(c)), "age_median_iqr": [int(q[0.5]), int(q[0.25]), int(q[0.75])],
            "female_pct": round(100 * float((c["gender"] == "F").mean()), 1),
            "mortality_pct": round(100 * float(c["hospital_expire_flag"].mean()), 1)}


def load(data_dir: Path) -> pd.DataFrame:
    cohort = pd.read_parquet(data_dir / "cohort.parquet")
    with open(data_dir / "predictions.pkl", "rb") as f:
        preds = pickle.load(f)
    cohort["p"] = cohort["hadm_id"].map(preds)
    cohort["y"] = cohort["hospital_expire_flag"].astype(int)
    return cohort.dropna(subset=["p"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mimic", type=Path, required=True)
    ap.add_argument("--eicu", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    m, e = load(args.mimic), load(args.eicu)
    res = {"performance": {"mimic_val": evaluate(m[m["split"] == "val"]),
                           "mimic_test": evaluate(m[m["split"] == "test"]),
                           "eicu": evaluate(e)},
           "demographics": {s: demographics(m[m["split"] == s]) for s in ("train", "val", "test")}}
    res["demographics"]["eicu"] = demographics(e)
    json.dump(res, open(args.out, "w"), indent=2)
    logger.info(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

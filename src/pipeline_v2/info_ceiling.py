"""Information-ceiling test (plan/info-ceiling-test.md; post hoc, specified before running).

Does the trajectory carry outcome information beyond the 209 aggregate
features? Compares XGBoost on aggregates (A) with aggregates plus
timeline-dynamics features (B), overall and within label strata.

Usage:
    python -m src.pipeline_v2.info_ceiling
"""

import json
import logging
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from xgboost import XGBClassifier

from .build import load_events
from .config import DATA_DIR, FEATURE_LABS, FEATURE_VITALS, MODEL

logger = logging.getLogger(__name__)
TAU, N_BOOT, SEED = 0.296, 1000, 42


def _slope(d: pd.DataFrame) -> pd.Series:
    d = d.assign(tv=d["t"] * d["value"], tt=d["t"] ** 2)
    s = d.groupby(["hadm_id", "label"])[["t", "value", "tv", "tt"]].sum()
    n = d.groupby(["hadm_id", "label"]).size()
    var_t = s["tt"] - s["t"] ** 2 / n
    cov = s["tv"] - s["t"] * s["value"] / n
    return (cov / var_t).where(var_t > 1e-9)


def dynamics_features(ev: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    labs = ev["lab"][ev["lab"]["label"].isin(FEATURE_LABS)]
    vit = ev["vital"][ev["vital"]["label"].isin(FEATURE_VITALS)]
    d = pd.concat([labs, vit])[["hadm_id", "label", "t", "value"]].dropna()
    d = d.sort_values(["hadm_id", "label", "t"]).reset_index(drop=True)
    g = d.groupby(["hadm_id", "label"])
    f = pd.DataFrame({
        "early_mean": d[d["t"] < 6].groupby(["hadm_id", "label"])["value"].mean(),
        "late_mean": d[d["t"] >= 18].groupby(["hadm_id", "label"])["value"].mean(),
        "accel": _slope(d[d["t"] >= 12]) - _slope(d[d["t"] < 12]),
        "volatility": d.assign(dv=g["value"].diff().abs()).groupby(["hadm_id", "label"])["dv"].mean(),
        "time_since_last": 24 - g["t"].max(),
        "t_of_max": d.loc[g["value"].idxmax(), ["hadm_id", "label", "t"]].set_index(["hadm_id", "label"])["t"],
        "t_of_min": d.loc[g["value"].idxmin(), ["hadm_id", "label", "t"]].set_index(["hadm_id", "label"])["t"],
        "n_last12": d[d["t"] >= 12].groupby(["hadm_id", "label"]).size(),
    })
    f["late_minus_early"] = f["late_mean"] - f["early_mean"]
    f["n_last12"] = f["n_last12"].fillna(0)
    wide = f.unstack("label")
    wide.columns = [f"{lab}_dyn_{stat}" for stat, lab in wide.columns]
    return wide


def fit(X: Dict[str, np.ndarray], y: Dict[str, np.ndarray]) -> XGBClassifier:
    m = XGBClassifier(n_estimators=MODEL.n_estimators, max_depth=MODEL.max_depth,
                      learning_rate=MODEL.learning_rate, subsample=MODEL.subsample,
                      colsample_bytree=MODEL.colsample_bytree, eval_metric="logloss",
                      early_stopping_rounds=MODEL.early_stopping_rounds, random_state=MODEL.seed,
                      n_jobs=-1, tree_method="hist")
    m.fit(X["train"], y["train"], eval_set=[(X["val"], y["val"])], verbose=False)
    return m


def paired_boot(y: np.ndarray, pa: np.ndarray, pb: np.ndarray, metric) -> Dict[str, object]:
    rng = np.random.default_rng(SEED)
    n = len(y)
    diffs = []
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        if y[i].min() == y[i].max():
            continue
        diffs.append(metric(y[i], pb[i]) - metric(y[i], pa[i]))
    return {"A": round(float(metric(y, pa)), 4), "B": round(float(metric(y, pb)), 4),
            "diff": round(float(metric(y, pb) - metric(y, pa)), 4),
            "ci95": np.percentile(diffs, [2.5, 97.5]).round(4).tolist()}


def main() -> None:
    feat = pd.read_parquet(DATA_DIR / "features.parquet").set_index("hadm_id")
    cohort = pd.read_parquet(DATA_DIR / "cohort.parquet")
    dyn = dynamics_features(load_events(cohort))
    full = feat.join(dyn)
    base_cols = [c for c in feat.columns if c not in ("mortality", "split")]
    dyn_cols = list(dyn.columns)
    logger.info(f"Aggregate features: {len(base_cols)}; dynamics features: {len(dyn_cols)}")
    parts = {s: full[full["split"] == s] for s in ("train", "val", "test")}
    y = {s: parts[s]["mortality"].values for s in parts}
    p = {}
    for name, cols in (("A", base_cols), ("B", base_cols + dyn_cols)):
        X = {s: parts[s][cols].values.astype(np.float32) for s in parts}
        m = fit(X, y)
        p[name] = {s: m.predict_proba(X[s])[:, 1] for s in parts}
        logger.info(f"Model {name}: best iteration {m.best_iteration}")

    yt, pa, pb = y["test"], p["A"]["test"], p["B"]["test"]
    res: Dict[str, object] = {
        "n_test": int(len(yt)), "n_dynamics_features": len(dyn_cols),
        "auroc": paired_boot(yt, pa, pb, roc_auc_score),
        "auprc": paired_boot(yt, pa, pb, average_precision_score),
        "brier": paired_boot(yt, pa, pb, brier_score_loss),
    }
    for stratum, mask in (("HIGH", pa >= TAU), ("LOW", pa < TAU)):
        res[f"within_{stratum}"] = {"n": int(mask.sum()), "deaths": int(yt[mask].sum()),
                                     **paired_boot(yt[mask], pa[mask], pb[mask], roc_auc_score)}
    lg = lambda x: np.log(np.clip(x, 1e-4, 1 - 1e-4) / (1 - np.clip(x, 1e-4, 1 - 1e-4)))
    stack = LogisticRegression(C=1e6).fit(np.column_stack([lg(p["A"]["val"]), lg(p["B"]["val"])]), y["val"])
    ps = stack.predict_proba(np.column_stack([lg(pa), lg(pb)]))[:, 1]
    res["stacked_A_plus_B"] = {"coef_logitA": round(float(stack.coef_[0][0]), 3),
                               "coef_logitB": round(float(stack.coef_[0][1]), 3),
                               "auroc": paired_boot(yt, pa, ps, roc_auc_score),
                               "brier": paired_boot(yt, pa, ps, brier_score_loss)}
    a = res["auroc"]
    res["decision"] = {
        "meaningful_overall": bool(a["diff"] >= 0.010 and a["ci95"][0] > 0),
        "meaningful_within_stratum": any(res[f"within_{s}"]["diff"] >= 0.02 and res[f"within_{s}"]["ci95"][0] > 0
                                         for s in ("HIGH", "LOW")),
    }
    json.dump(res, open(DATA_DIR / "info_ceiling.json", "w"), indent=2)
    logger.info(json.dumps(res, indent=1))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()

"""Train the base mortality model and fix its decision threshold on validation.

No class reweighting is used, so predicted probabilities stay on the outcome
scale. The HIGH/LOW risk label shown to explanation methods and judges uses
the threshold that maximizes F1 on the validation split.

Usage:
    python -m src.pipeline_v2.model
"""

import json
import logging
import pickle
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             precision_recall_curve, roc_auc_score)
from xgboost import XGBClassifier

from .config import DATA_DIR, MODEL

logger = logging.getLogger(__name__)
NON_FEATURES = ("hadm_id", "mortality", "split")


def _calibration_slope_intercept(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    """Logistic recalibration of outcome on logit(p): slope 1 and intercept 0 are ideal."""
    logit = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
    lr = LogisticRegression(C=1e6).fit(logit.reshape(-1, 1), y)
    return {"cal_slope": float(lr.coef_[0][0]), "cal_intercept": float(lr.intercept_[0])}


def _metrics(y: np.ndarray, p: np.ndarray, thr: float) -> Dict[str, float]:
    pred = p >= thr
    tp = int((pred & (y == 1)).sum())
    out = {
        "n": int(len(y)), "prevalence": float(y.mean()),
        "auroc": float(roc_auc_score(y, p)), "auprc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)), "brier_null": float(y.mean() * (1 - y.mean())),
        "sensitivity": tp / max(int((y == 1).sum()), 1),
        "ppv": tp / max(int(pred.sum()), 1),
        "flagged_frac": float(pred.mean()),
    }
    out.update(_calibration_slope_intercept(y, p))
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in out.items()}


def train() -> Dict[str, object]:
    df = pd.read_parquet(DATA_DIR / "features.parquet")
    cols = [c for c in df.columns if c not in NON_FEATURES]
    parts = {s: df[df["split"] == s] for s in ("train", "val", "test")}
    imputer = SimpleImputer(strategy="median")
    X = {s: (imputer.fit_transform if s == "train" else imputer.transform)(
        parts[s][cols].values.astype(np.float32)) for s in ("train", "val", "test")}
    y = {s: parts[s]["mortality"].values for s in parts}

    model = XGBClassifier(
        n_estimators=MODEL.n_estimators, max_depth=MODEL.max_depth,
        learning_rate=MODEL.learning_rate, subsample=MODEL.subsample,
        colsample_bytree=MODEL.colsample_bytree, eval_metric="logloss",
        early_stopping_rounds=MODEL.early_stopping_rounds,
        random_state=MODEL.seed, n_jobs=-1, tree_method="hist",
    )
    model.fit(X["train"], y["train"], eval_set=[(X["val"], y["val"])], verbose=100)

    p = {s: model.predict_proba(X[s])[:, 1] for s in X}
    prec, rec, thr = precision_recall_curve(y["val"], p["val"])
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-9, None)
    threshold = float(thr[int(np.argmax(f1))])
    logger.info(f"Decision threshold (max F1 on validation): {threshold:.4f}")

    results = {"n_features": len(cols), "best_iteration": int(model.best_iteration),
               "threshold": round(threshold, 4),
               "val": _metrics(y["val"], p["val"], threshold),
               "test": _metrics(y["test"], p["test"], threshold)}
    frac, mean_p = calibration_curve(y["test"], p["test"], n_bins=10, strategy="quantile")
    results["test_calibration_bins"] = {"mean_pred": mean_p.round(4).tolist(),
                                       "frac_pos": frac.round(4).tolist()}
    for s in ("val", "test"):
        logger.info(f"{s}: {results[s]}")

    (DATA_DIR / "models").mkdir(exist_ok=True)
    joblib.dump({"model": model, "imputer": imputer, "feature_cols": cols, "threshold": threshold},
                DATA_DIR / "models" / "xgboost_v2.pkl")
    preds = {int(h): float(v) for s in X for h, v in zip(parts[s]["hadm_id"], p[s])}
    with open(DATA_DIR / "predictions.pkl", "wb") as f:
        pickle.dump(preds, f)
    with open(DATA_DIR / "models" / "base_model_results.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    train()

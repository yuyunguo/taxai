"""Cohort description (Table 1) and base-model calibration figure.

Usage:
    python -m src.pipeline_v2.describe
"""

import json
import logging
import pickle
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve

from .config import DATA_DIR

logger = logging.getLogger(__name__)


def _median_iqr(s: pd.Series) -> str:
    q = s.quantile([0.25, 0.5, 0.75])
    return f"{q[0.5]:.0f} ({q[0.25]:.0f}-{q[0.75]:.0f})"


def _pct(mask: pd.Series) -> str:
    return f"{mask.sum():,} ({100 * mask.mean():.1f}%)"


def table1() -> Dict[str, Dict[str, str]]:
    cohort = pd.read_parquet(DATA_DIR / "cohort.parquet")
    feat = pd.read_parquet(DATA_DIR / "features.parquet").set_index("hadm_id")
    cohort = cohort.join(feat[["count_lab", "count_vital", "count_med", "count_note",
                               "Lactate_count", "Troponin_T_count", "hours_admit_to_icu"]], on="hadm_id")
    meta = json.load(open(DATA_DIR / "explanations" / "sample_meta.json"))
    sample_ids = {c["hadm_id"] for c in meta["cases"]}
    groups = {s: cohort[cohort["split"] == s] for s in ("train", "val", "test")}
    groups["explanation sample"] = cohort[cohort["hadm_id"].isin(sample_ids)]
    out: Dict[str, Dict[str, str]] = {}
    for name, d in groups.items():
        units = d["first_careunit"].value_counts(normalize=True).head(4)
        out[name] = {
            "n": f"{len(d):,}",
            "years (anchor group)": ", ".join(sorted(d["anchor_year_group"].unique())),
            "age, median (IQR)": _median_iqr(d["age"]),
            "female": _pct(d["gender"] == "F"),
            "emergency/urgent admission": _pct(d["admission_type"].str.contains("EMER|URGENT")),
            "hours admission to ICU, median (IQR)": _median_iqr(d["hours_admit_to_icu"]),
            "top ICU units": "; ".join(f"{u} {100 * v:.0f}%" for u, v in units.items()),
            "lab events, median (IQR)": _median_iqr(d["count_lab"]),
            "vital events, median (IQR)": _median_iqr(d["count_vital"]),
            "medication orders, median (IQR)": _median_iqr(d["count_med"]),
            "any lactate measured": _pct(d["Lactate_count"] > 0),
            "any troponin T measured": _pct(d["Troponin_T_count"] > 0),
            "in-hospital mortality": _pct(d["hospital_expire_flag"] == 1),
        }
    return out


def calibration_figure() -> None:
    feat = pd.read_parquet(DATA_DIR / "features.parquet")
    with open(DATA_DIR / "predictions.pkl", "rb") as f:
        preds = pickle.load(f)
    res = json.load(open(DATA_DIR / "models" / "base_model_results.json"))
    fig, ax = plt.subplots(figsize=(3.5, 3.4), dpi=300)
    for split, color in (("val", "#4C72B0"), ("test", "#C44E52")):
        d = feat[feat["split"] == split]
        p = d["hadm_id"].map(preds).values
        frac, mean_p = calibration_curve(d["mortality"].values, p, n_bins=10, strategy="quantile")
        label = (f"{'Validation' if split == 'val' else 'Test'} (AUROC {res[split]['auroc']:.3f}, "
                 f"Brier {res[split]['brier']:.3f})")
        ax.plot(mean_p, frac, "o-", ms=3, lw=1.2, color=color, label=label)
    ax.plot([0, 0.7], [0, 0.7], "--", color="grey", lw=0.8)
    ax.axvline(res["threshold"], color="black", lw=0.6, ls=":")
    ax.text(res["threshold"] + 0.01, 0.15, f"threshold {res['threshold']:.2f}", fontsize=6)
    ax.set_xlabel("Predicted mortality risk", fontsize=8)
    ax.set_ylabel("Observed mortality", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, frameon=False, loc="upper left")
    ax.set_xlim(0, 0.7)
    ax.set_ylim(0, 0.7)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    out = DATA_DIR / "figures"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "calibration_v2.png", bbox_inches="tight")
    logger.info(f"Saved {out / 'calibration_v2.png'}")


def main() -> None:
    t1 = table1()
    with open(DATA_DIR / "table1.json", "w") as f:
        json.dump(t1, f, indent=2)
    logger.info(json.dumps(t1, indent=1))
    calibration_figure()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()

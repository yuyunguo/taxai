"""Build model features and LLM-facing trajectories from extracted parquet files.

Times are expressed in hours relative to ICU admission (negative = before
ICU admission); prediction time is +24 h. Features use every in-window
measurement; the LLM timeline is a compressed, capped view of the same data.

Usage:
    python -m src.pipeline_v2.build
"""

import logging
import pickle
from typing import Dict, List

import numpy as np
import pandas as pd

from .config import (DATA_DIR, EVENT_TYPES, FEATURE_LABS, FEATURE_VITALS,
                     LAB_ITEMIDS, TRAJ, VITAL_ITEMIDS)

logger = logging.getLogger(__name__)

PLAUSIBLE = {
    "Heart_Rate": (0, 300), "Respiratory_Rate": (0, 80), "SpO2": (0, 100),
    "Temperature_C": (25, 45),
}
BP_RANGE = (0, 300)
CAREUNITS = ("Cardiac Vascular Intensive Care Unit (CVICU)", "Coronary Care Unit (CCU)",
             "Medical Intensive Care Unit (MICU)", "Surgical Intensive Care Unit (SICU)",
             "Medical/Surgical Intensive Care Unit (MICU/SICU)", "Trauma SICU (TSICU)",
             "Neuro Surgical Intensive Care Unit (Neuro SICU)")


def _relative_hours(df: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
    intime = cohort.set_index("hadm_id")["intime"]
    out = df.copy()
    out["t"] = (out["charttime"] - out["hadm_id"].map(intime)).dt.total_seconds() / 3600.0
    return out


def load_events(cohort: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Load and normalize labs, vitals, meds, and radiology references."""
    labs = pd.read_parquet(DATA_DIR / "labs.parquet")
    labs["label"] = labs["itemid"].map(LAB_ITEMIDS)

    vit = pd.read_parquet(DATA_DIR / "vitals.parquet")
    vit["label"] = vit["itemid"].map(VITAL_ITEMIDS)
    is_f = vit["label"] == "Temperature_F"
    vit.loc[is_f, "value"] = (vit.loc[is_f, "value"] - 32) * 5 / 9
    vit.loc[is_f, "label"] = "Temperature_C"
    lo = vit["label"].map(lambda x: PLAUSIBLE.get(x, BP_RANGE)[0])
    hi = vit["label"].map(lambda x: PLAUSIBLE.get(x, BP_RANGE)[1])
    vit = vit[(vit["value"] >= lo) & (vit["value"] <= hi)]

    meds = pd.read_parquet(DATA_DIR / "meds.parquet")
    meds["label"] = meds["drug"].str.strip().str[:60]
    notes = pd.read_parquet(DATA_DIR / "notes.parquet")
    notes["label"] = "Radiology report"

    events = {"lab": labs, "vital": vit, "med": meds, "note": notes}
    return {k: _relative_hours(v, cohort) for k, v in events.items()}


def _series_features(df: pd.DataFrame, labels: tuple) -> pd.DataFrame:
    """Eight summary statistics per (admission, variable), vectorized."""
    d = df[df["label"].isin(labels)].sort_values(["hadm_id", "label", "t"])
    g = d.groupby(["hadm_id", "label"])
    stats = g["value"].agg(["mean", "std", "min", "max", "first", "last", "count"])
    d = d.assign(tv=d["t"] * d["value"], tt=d["t"] ** 2)
    s = d.groupby(["hadm_id", "label"])[["t", "value", "tv", "tt"]].sum()
    n = stats["count"]
    var_t = s["tt"] - s["t"] ** 2 / n
    cov = s["tv"] - s["t"] * s["value"] / n
    stats["trend"] = np.where(var_t > 1e-9, cov / var_t.where(var_t > 1e-9, 1), 0.0)
    stats["std"] = stats["std"].fillna(0.0)
    wide = stats.unstack("label")
    wide.columns = [f"{lab}_{stat}" for stat, lab in wide.columns]
    for lab in labels:
        cnt = f"{lab}_count"
        if cnt not in wide:
            wide[cnt] = 0
        wide[cnt] = wide[cnt].fillna(0)
    return wide


def build_features(cohort: pd.DataFrame, ev: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Assemble the model feature matrix (no ICD codes, prediction-time information only)."""
    base = cohort.set_index("hadm_id")
    feat = pd.DataFrame(index=base.index)
    feat["age"] = base["age"].astype(float)
    feat["male"] = (base["gender"] == "M").astype(int)
    feat["emergency"] = base["admission_type"].str.contains("EMER|URGENT", regex=True).astype(int)
    feat["hours_admit_to_icu"] = (base["intime"] - base["admittime"]).dt.total_seconds() / 3600.0
    for unit in CAREUNITS:
        feat[f"unit_{unit.split('(')[-1].rstrip(')')}"] = (base["first_careunit"] == unit).astype(int)
    feat = feat.join(_series_features(ev["lab"], FEATURE_LABS))
    feat = feat.join(_series_features(ev["vital"], FEATURE_VITALS))
    for etype in EVENT_TYPES:
        feat[f"count_{etype}"] = ev[etype].groupby("hadm_id").size().reindex(feat.index).fillna(0)
    feat["n_unique_meds"] = ev["med"].groupby("hadm_id")["label"].nunique().reindex(feat.index).fillna(0)
    feat["mortality"] = base["hospital_expire_flag"].astype(int)
    feat["split"] = base["split"]
    return feat.reset_index()


def _vital_view(vit: pd.DataFrame) -> pd.DataFrame:
    """Map vitals to timeline variables, preferring arterial over non-invasive pressure."""
    has_art = (vit["label"] == "Arterial_BP_Mean").any()
    src = "Arterial_BP_" if has_art else "Non_Invasive_BP_"
    mapping = {"Heart_Rate": "Heart_Rate", "Respiratory_Rate": "Respiratory_Rate", "SpO2": "SpO2",
               f"{src}Systolic": "SBP", f"{src}Mean": "MAP"}
    return vit.assign(label=vit["label"].map(mapping)).dropna(subset=["label"])


def _timeline_events(hid: int, ev: Dict[str, pd.DataFrame]) -> List[dict]:
    """Compressed chronological timeline for one admission (LLM input), with per-type budgets."""
    budget = dict(TRAJ.budget)
    prio = {t: i for i, t in enumerate(TRAJ.type_priority)}
    rows: List[dict] = []

    def group(kind: str):
        return ev[kind].get_group(hid) if hid in ev[kind].groups else None

    lab = group("lab")
    if lab is not None:
        # PO2 specimen type (arterial vs venous) is unreliable, so it is withheld from the LLM view.
        lab = lab[lab["label"] != "PO2"].sort_values("t")
        keep = pd.concat([lab.groupby("label").head(1), lab.groupby("label").tail(1)]).drop_duplicates()
        keep = keep.sort_values("t", ascending=False).head(budget["lab"])  # prefer recent values
        rows += [{"type": "lab", "t": float(r.t), "label": r.label, "value": round(float(r.value), 2),
                  "uom": r.uom or ""} for r in keep.itertuples()]
    vit = group("vital")
    if vit is not None:
        vit = _vital_view(vit)
        vit = vit.assign(b=np.floor(vit["t"] / TRAJ.vital_bin_h) * TRAJ.vital_bin_h)
        agg = vit.groupby(["label", "b"])["value"].median().reset_index()
        agg = agg.sort_values("b", ascending=False).head(budget["vital"])
        rows += [{"type": "vital", "t": float(r.b), "label": r.label, "value": round(float(r.value), 1),
                  "uom": ""} for r in agg.itertuples()]
    med = group("med")
    if med is not None:
        med = med.sort_values("t").drop_duplicates("label").head(budget["med"])
        rows += [{"type": "med", "t": float(r.t), "label": r.label, "value": None, "uom": ""}
                 for r in med.itertuples()]
    note = group("note")
    if note is not None:
        note = note.sort_values("t").tail(budget["note"])
        rows += [{"type": "note", "t": float(r.t), "label": f"Radiology report ({r.note_type})",
                  "value": None, "uom": ""} for r in note.itertuples()]
    return sorted(rows, key=lambda r: (r["t"], prio[r["type"]]))


def build_trajectories(cohort: pd.DataFrame, ev: Dict[str, pd.DataFrame], hids: List[int]) -> Dict[int, dict]:
    """Timelines plus static context (known at prediction time) for the given admissions."""
    grouped = {k: v[v["hadm_id"].isin(hids)].groupby("hadm_id") for k, v in ev.items()}
    base = cohort.set_index("hadm_id")
    out: Dict[int, dict] = {}
    for hid in hids:
        b = base.loc[hid]
        out[int(hid)] = {
            "hadm_id": int(hid),
            "context": {"age": int(b["age"]), "sex": b["gender"],
                        "admission_type": b["admission_type"], "icu_unit": b["first_careunit"]},
            "events": _timeline_events(hid, grouped),
            "mortality": int(b["hospital_expire_flag"]),
            "split": b["split"],
        }
    return out


def main() -> None:
    cohort = pd.read_parquet(DATA_DIR / "cohort.parquet")
    ev = load_events(cohort)
    feat = build_features(cohort, ev)
    feat.to_parquet(DATA_DIR / "features.parquet", index=False)
    n_feat = feat.shape[1] - 3
    logger.info(f"Features: {feat.shape[0]:,} admissions x {n_feat} features")
    test_ids = cohort.loc[cohort["split"] == "test", "hadm_id"].astype(int).tolist()
    trajs = build_trajectories(cohort, ev, test_ids)
    with open(DATA_DIR / "trajectories_test.pkl", "wb") as f:
        pickle.dump(trajs, f)
    lens = [len(t["events"]) for t in trajs.values()]
    logger.info(f"Test trajectories: {len(trajs):,}; events mean {np.mean(lens):.1f}, "
                f"median {np.median(lens):.0f}, max {np.max(lens)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()

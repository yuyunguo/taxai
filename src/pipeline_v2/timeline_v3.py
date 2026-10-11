"""v3 chart: the full chronological record used as the judge's chart in every
arm and as TAXAI's generation input.

Compared with the v2 timeline (first and last lab values, 8-h vital medians),
v3 keeps every in-window laboratory value (one per test per hour), 2-hour
vital-sign medians including temperature, the first 15 medication starts,
and all imaging references. PO2 remains withheld (unreliable specimen type).

Usage:
    python -m src.pipeline_v2.timeline_v3 [--split test|val]
"""

import argparse
import logging
import pickle
from typing import Dict, List

import numpy as np
import pandas as pd

from .build import load_events
from .config import DATA_DIR

logger = logging.getLogger(__name__)
VITAL_BIN_H = 2
MAX_MEDS = 15
PRIO = {"lab": 0, "vital": 1, "med": 2, "note": 3}


def _vitals(vit: pd.DataFrame) -> pd.DataFrame:
    has_art = (vit["label"] == "Arterial_BP_Mean").any()
    src = "Arterial_BP_" if has_art else "Non_Invasive_BP_"
    mapping = {"Heart_Rate": "Heart_Rate", "Respiratory_Rate": "Respiratory_Rate", "SpO2": "SpO2",
               "Temperature_C": "Temperature_C", f"{src}Systolic": "SBP", f"{src}Mean": "MAP"}
    v = vit.assign(label=vit["label"].map(mapping)).dropna(subset=["label"])
    v = v.assign(b=np.floor(v["t"] / VITAL_BIN_H) * VITAL_BIN_H)
    return v.groupby(["label", "b"])["value"].median().reset_index()


def chart_events(hid: int, g: Dict[str, "pd.core.groupby.DataFrameGroupBy"]) -> List[dict]:
    rows: List[dict] = []
    if hid in g["lab"].groups:
        lab = g["lab"].get_group(hid)
        lab = lab[lab["label"] != "PO2"].assign(h=lambda d: np.floor(d["t"]))
        lab = lab.sort_values("t").drop_duplicates(["label", "h"], keep="last")
        rows += [{"type": "lab", "t": float(r.t), "label": r.label, "value": round(float(r.value), 2),
                  "uom": r.uom or ""} for r in lab.itertuples()]
    if hid in g["vital"].groups:
        rows += [{"type": "vital", "t": float(r.b), "label": r.label, "value": round(float(r.value), 1),
                  "uom": ""} for r in _vitals(g["vital"].get_group(hid)).itertuples()]
    if hid in g["med"].groups:
        med = g["med"].get_group(hid).sort_values("t").drop_duplicates("label").head(MAX_MEDS)
        rows += [{"type": "med", "t": float(r.t), "label": r.label, "value": None, "uom": ""}
                 for r in med.itertuples()]
    if hid in g["note"].groups:
        rows += [{"type": "note", "t": float(r.t), "label": f"Radiology report ({r.note_type})",
                  "value": None, "uom": ""} for r in g["note"].get_group(hid).itertuples()]
    return sorted(rows, key=lambda r: (r["t"], PRIO[r["type"]]))


def main(split: str = "test") -> None:
    cohort = pd.read_parquet(DATA_DIR / "cohort.parquet")
    test = cohort[cohort["split"] == split].set_index("hadm_id")
    ev = load_events(cohort)
    ids = test.index.astype(int).tolist()
    g = {k: v[v["hadm_id"].isin(ids)].groupby("hadm_id") for k, v in ev.items()}
    out = {}
    for hid in ids:
        b = test.loc[hid]
        out[hid] = {"hadm_id": hid,
                    "context": {"age": int(b["age"]), "sex": b["gender"],
                                "admission_type": b["admission_type"], "icu_unit": b["first_careunit"]},
                    "events": chart_events(hid, g), "mortality": int(b["hospital_expire_flag"]),
                    "split": split}
    with open(DATA_DIR / f"trajectories_{split}_v3.pkl", "wb") as f:
        pickle.dump(out, f)
    n = [len(t["events"]) for t in out.values()]
    logger.info(f"v3 charts: {len(out):,}; events median {np.median(n):.0f} (IQR "
                f"{np.percentile(n, 25):.0f}-{np.percentile(n, 75):.0f}), max {max(n)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=("test", "val"))
    main(ap.parse_args().split)

"""Trajectory-citation verification: are the numbers an explanation cites in the data?

For each numeric value in an explanation that can be attributed to a clinical
variable (by the nearest preceding variable mention in the same sentence),
the value is *supported* if it matches, within rounding tolerance, either a
recorded value of that variable for the same patient or a difference between
two recorded values (stated changes). Numbers that cannot be attributed to a
variable (times, percentages of the prediction, ages) are counted separately
and excluded from the rate.

Pre-specified as a secondary endpoint in plan/analysis-plan-v2.md (H4).

Usage:
    python -m src.pipeline_v2.faithfulness
"""

import json
import logging
import pickle
import re
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .config import DATA_DIR

logger = logging.getLogger(__name__)

ALIASES: Dict[str, Tuple[str, ...]] = {
    "Lactate": ("lactate", "lactic acid"),
    "Creatinine": ("creatinine", "cr"),
    "Troponin_T": ("troponin t", "troponin", "trop t", "tnt"),
    "Troponin_I": ("troponin i", "tni"),
    "NTproBNP": ("nt-probnp", "ntprobnp", "bnp"),
    "WBC": ("wbc", "white blood cell", "white cell", "leukocyte"),
    "Potassium": ("potassium", "k+"),
    "Sodium": ("sodium", "na+"),
    "Glucose": ("glucose", "blood sugar"),
    "Hemoglobin": ("hemoglobin", "hgb", "hb"),
    "Platelets": ("platelet", "plt"),
    "Chloride": ("chloride",),
    "Bicarbonate": ("bicarbonate", "hco3", "bicarb"),
    "BUN": ("bun", "urea nitrogen", "blood urea"),
    "Calcium_Total": ("calcium",),
    "Calculated_Total_CO2": ("total co2", "tco2"),
    "pH": ("ph",),
    "pCO2": ("pco2", "paco2"),
    "Base_Excess": ("base excess", "base deficit"),
    "Heart_Rate": ("heart rate", "hr", "pulse"),
    "SBP": ("systolic", "sbp"),
    "MAP": ("mean arterial", "map"),
    "Respiratory_Rate": ("respiratory rate", "rr", "breathing rate", "respirations"),
    "SpO2": ("spo2", "oxygen saturation", "o2 sat", "saturation", "sao2"),
}
_ALIAS_RE = re.compile(
    r"(?<![a-z0-9])(" + "|".join(sorted({re.escape(a) for v in ALIASES.values() for a in v},
                                       key=len, reverse=True)) + r")s?(?![a-z0-9])")
_ALIAS_TO_VAR = {a: var for var, al in ALIASES.items() for a in al}
_NUM_RE = re.compile(r"(?<![\w.])[-+]?\d+(?:\.\d+)?")
_SKIP_AFTER = re.compile(r"^\s*-?\s*(?:%|h\b|hr\b|hrs\b|hour|hours|year|yo\b|day|min)")
_SKIP_BEFORE = re.compile(r"(?:hour|hours|icu hour|day|age|aged)\s*[+-]?\s*$")
MAX_GAP = 60  # characters between variable mention and number


def _sentences(text: str) -> List[str]:
    return [s for s in re.split(r"(?<=[.;!?])\s+|\n+", text) if s.strip()]


def extract_claims(text: str) -> Tuple[List[Tuple[str, float, Tuple[str, ...]]], int]:
    """Return (variable, value, alternative variables) claims and the count of unattributed numbers."""
    claims: List[Tuple[str, float]] = []
    unattributed = 0
    for sent in _sentences(text.lower().replace("_", " ")):
        mentions = [(m.end(), _ALIAS_TO_VAR[m.group(1)]) for m in _ALIAS_RE.finditer(sent)]
        for m in _NUM_RE.finditer(sent):
            if _SKIP_AFTER.match(sent[m.end():]) or _SKIP_BEFORE.search(sent[:m.start()]):
                continue
            prior = [(end, var) for end, var in mentions if end <= m.start() and m.start() - end <= MAX_GAP]
            if prior:
                # Primary attribution is the nearest preceding variable; the other variables
                # named earlier in the sentence are kept as alternatives ("X and Y were a and b").
                alts = tuple(dict.fromkeys(v for e, v in mentions if e <= m.start()))
                claims.append((prior[-1][1], float(m.group()), alts))
            else:
                unattributed += 1
    return claims, unattributed


def patient_values(traj: dict) -> Dict[str, List[float]]:
    vals: Dict[str, List[float]] = {}
    for e in traj["events"]:
        if e["type"] in ("lab", "vital") and e["value"] is not None:
            vals.setdefault(e["label"], []).append(float(e["value"]))
    return vals


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= max(0.051, 0.011 * abs(b))


def is_supported(var: str, value: float, vals: Dict[str, List[float]]) -> bool:
    recorded = vals.get(var, [])
    if any(_close(value, v) for v in recorded):
        return True
    return any(_close(abs(value), abs(a - b)) for a, b in combinations(recorded, 2))


def check(text: str, traj: dict) -> Dict[str, object]:
    claims, unattributed = extract_claims(text)
    vals = patient_values(traj)
    unsupported = [(v, x) for v, x, alts in claims
                   if not any(is_supported(a, x, vals) for a in (v,) + alts)]
    n = len(claims)
    return {"n_claims": n, "n_supported": n - len(unsupported), "n_unattributed": unattributed,
            "rate": (n - len(unsupported)) / n if n else None,
            "unsupported": [f"{v}={x:g}" for v, x in unsupported]}


def main() -> None:
    exp_dir = DATA_DIR / "explanations"
    expl = json.load(open(exp_dir / "explanations.json"))
    with open(DATA_DIR / "trajectories_test.pkl", "rb") as f:
        trajs = pickle.load(f)
    rows = []
    for method in ("random", "static_llm", "taxai"):
        for hid, text in expl[method].items():
            r = check(text, trajs[int(hid)])
            rows.append({"hadm_id": int(hid), "method": method, **r,
                         "unsupported": "; ".join(r["unsupported"])})
    df = pd.DataFrame(rows)
    out = DATA_DIR / "cuef_v2"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "faithfulness_per_case.csv", index=False)
    for m, d in df.groupby("method"):
        tot, sup = d["n_claims"].sum(), d["n_supported"].sum()
        logger.info(f"{m:10s} claims={tot} supported={sup} pooled_rate={sup / max(tot, 1):.3f} "
                    f"explanations_with_unsupported={(d['n_claims'] > d['n_supported']).mean():.3f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()

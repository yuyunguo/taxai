"""Value and temporal-claim faithfulness of v3 explanations against the v3 chart.

A value claim is a number attributed to a clinical variable (same rules as
faithfulness.py). A temporal claim is a value claim with an "hour N" mention in the same
sentence (preferring one up to 20 characters after the value, else the
nearest one up to 40 characters before it); it is supported if the patient
has that variable at that value within +/-1 h (labs) or +/-2 h (2-hour vital
bins) of the cited ICU hour.

Usage:
    python -m src.pipeline_v2.faithfulness_v3 --set main
"""

import argparse
import json
import logging
import pickle
import re
from typing import Dict, List, Tuple

import pandas as pd

from .config import DATA_DIR
from .faithfulness import (_ALIAS_RE, _ALIAS_TO_VAR, _NUM_RE, _SKIP_AFTER, _SKIP_BEFORE,
                           MAX_GAP, _close, _sentences, is_supported, patient_values)

logger = logging.getLogger(__name__)
_HOUR = re.compile(r"hour\s*([+-]?\d+)")
HOUR_WINDOW = 40
FOLLOW_GAP = 20
TOL_H = {"lab": 1.0, "vital": 2.0}


def claims_with_hours(text: str) -> List[Tuple[str, float, Tuple[str, ...], float]]:
    """(variable, value, alternative variables, cited hour or NaN) for each attributed number."""
    out = []
    for sent in _sentences(text.lower().replace("_", " ")):
        mentions = [(m.end(), _ALIAS_TO_VAR[m.group(1)]) for m in _ALIAS_RE.finditer(sent)]
        hours = [(m.start(), float(m.group(1))) for m in _HOUR.finditer(sent)]
        hour_spans = [(m.start(), m.end()) for m in _HOUR.finditer(sent)]
        for m in _NUM_RE.finditer(sent):
            if any(a <= m.start() < b for a, b in hour_spans):
                continue
            if _SKIP_AFTER.match(sent[m.end():]) or _SKIP_BEFORE.search(sent[:m.start()]):
                continue
            prior = [(e, v) for e, v in mentions if e <= m.start() and m.start() - e <= MAX_GAP]
            if not prior:
                continue
            alts = tuple(dict.fromkeys(v for e, v in mentions if e <= m.start()))
            hour = _cited_hour(m.start(), m.end(), hour_spans, hours)
            out.append((prior[-1][1], float(m.group()), alts, hour))
    return out


def _cited_hour(start: int, end: int, spans: List[Tuple[int, int]], hours: List[Tuple[int, float]]) -> float:
    """Prefer an hour mention just after the value ("4.8 at hour 15"), else the nearest preceding one."""
    after = [(a - end, h) for (a, _), (_, h) in zip(spans, hours) if 0 <= a - end <= FOLLOW_GAP]
    if after:
        return min(after)[1]
    before = [(start - b, h) for (_, b), (_, h) in zip(spans, hours) if 0 <= start - b <= HOUR_WINDOW]
    return min(before)[1] if before else float("nan")


def temporal_supported(var: str, value: float, hour: float, traj: dict) -> bool:
    for e in traj["events"]:
        if e["type"] in TOL_H and e["label"] == var and e["value"] is not None:
            if _close(value, float(e["value"])) and abs(e["t"] - hour) <= TOL_H[e["type"]]:
                return True
    return False


def check(text: str, traj: dict) -> Dict[str, int]:
    vals = patient_values(traj)
    c = claims_with_hours(text)
    val_ok = [any(is_supported(a, x, vals) for a in (v,) + alts) for v, x, alts, _ in c]
    temporal = [(v, x, alts, h) for v, x, alts, h in c if h == h]
    t_ok = [any(temporal_supported(a, x, h, traj) for a in (v,) + alts) for v, x, alts, h in temporal]
    return {"n_value_claims": len(c), "n_value_supported": sum(val_ok),
            "n_temporal_claims": len(temporal), "n_temporal_supported": sum(t_ok)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["pilot", "main"], required=True)
    args = ap.parse_args()
    od = DATA_DIR / "v3" / args.set
    expl = json.load(open(od / "explanations.json"))
    with open(DATA_DIR / "trajectories_test_v3.pkl", "rb") as f:
        trajs = pickle.load(f)
    rows = [{"hadm_id": int(h), "method": m, **check(t, trajs[int(h)])}
            for m in ("random", "static_llm", "taxai") for h, t in expl[m].items()]
    df = pd.DataFrame(rows)
    df.to_csv(od / "faithfulness_v3_per_case.csv", index=False)
    for m, d in df.groupby("method"):
        logger.info(f"{m:10s} value {d.n_value_supported.sum()}/{d.n_value_claims.sum()} "
                    f"temporal {d.n_temporal_supported.sum()}/{d.n_temporal_claims.sum()} "
                    f"(explanations with >=1 temporal claim: {(d.n_temporal_claims > 0).mean():.2f})")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()

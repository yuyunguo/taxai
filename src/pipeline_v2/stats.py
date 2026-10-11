"""Statistics for CUEF v2 from per-case judge outputs.

Primary decision endpoint: balanced appropriate reliance
    BAR = 0.5 * [P(accept | label correct) + P(override | label wrong)],
where the HIGH/LOW label is correct if it matches the in-hospital outcome.
CU = 0.4 * DC' + 0.3 * TAS + 0.3 * BAR. The v1 composite with (1 - OR) is
reported alongside for comparison.

Usage:
    python -m src.pipeline_v2.stats
"""

import json
import logging
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.contingency_tables import mcnemar

from .config import DATA_DIR, STUDY

logger = logging.getLogger(__name__)
METHODS = ["empty", "random", "shap", "static_llm", "taxai"]
N_BOOT = 2000


def _bar(override: np.ndarray, correct: np.ndarray) -> float:
    return 0.5 * ((1 - override[correct]).mean() + override[~correct].mean())


def summarize(df: pd.DataFrame) -> Dict[str, dict]:
    a, b, g = STUDY.cu_weights
    out = {}
    for m in METHODS:
        d = df[df["method"] == m]
        o, c = d["override"].values, d["correct"].values
        bar = _bar(o, c)
        out[m] = {"n": len(d), "DC": d["dc"].mean(), "TAS": d["tas"].mean(), "OR": o.mean(),
                  "accept_if_correct": (1 - o[c]).mean(), "override_if_wrong": o[~c].mean(),
                  "BAR": bar, "CU": a * d["dc"].mean() + b * d["tas"].mean() + g * bar,
                  "CU_v1_style": a * d["dc"].mean() + b * d["tas"].mean() + g * (1 - o.mean())}
    return {m: {k: round(float(v), 4) for k, v in r.items()} for m, r in out.items()}


def _paired_bootstrap(wide: Dict[str, pd.DataFrame], other: str, seed: int) -> Dict[str, dict]:
    """Case-resampled CIs for TAXAI minus `other` on BAR and CU."""
    a, b, g = STUDY.cu_weights
    t, o = wide["taxai"], wide[other]
    corr = t["correct"].values
    rng = np.random.default_rng(seed)
    n = len(t)
    res = {"BAR": [], "CU": []}
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        bt, bo = _bar(t["override"].values[i], corr[i]), _bar(o["override"].values[i], corr[i])
        res["BAR"].append(bt - bo)
        cu_t = a * t["dc"].values[i].mean() + b * t["tas"].values[i].mean() + g * bt
        cu_o = a * o["dc"].values[i].mean() + b * o["tas"].values[i].mean() + g * bo
        res["CU"].append(cu_t - cu_o)
    out = {}
    for k, v in res.items():
        v = np.array(v)
        p = 2 * min((v <= 0).mean(), (v >= 0).mean())
        out[k] = {"diff": round(float(np.mean(v)), 4),
                  "ci95": [round(float(x), 4) for x in np.percentile(v, [2.5, 97.5])],
                  "p_boot": round(float(max(p, 1 / N_BOOT)), 4)}
    return out


def pairwise_tests(df: pd.DataFrame) -> Dict[str, dict]:
    wide = {m: df[df["method"] == m].sort_values("hadm_id").reset_index(drop=True) for m in METHODS}
    n_tests = 4 * 5  # 4 comparisons x (DC, TAS, OR, BAR, CU)
    out = {}
    for other in METHODS[:-1]:
        t, o = wide["taxai"], wide[other]
        r = {}
        for metric in ("dc", "tas"):
            diff = t[metric].values - o[metric].values
            p = wilcoxon(t[metric], o[metric]).pvalue if np.any(diff != 0) else 1.0
            r[metric.upper()] = {"diff": round(float(diff.mean()), 4), "p": float(p),
                                 "p_bonf": min(1.0, float(p) * n_tests)}
        b_ = int(((t["override"] == 1) & (o["override"] == 0)).sum())
        c_ = int(((t["override"] == 0) & (o["override"] == 1)).sum())
        p = mcnemar([[0, b_], [c_, 0]], exact=False, correction=True).pvalue if b_ + c_ else 1.0
        r["OR"] = {"diff": round(float(t["override"].mean() - o["override"].mean()), 4),
                   "p": float(p), "p_bonf": min(1.0, float(p) * n_tests)}
        for k, v in _paired_bootstrap(wide, other, STUDY.seed).items():
            v["p_bonf"] = min(1.0, v["p_boot"] * n_tests)
            r[k] = v
        out[f"taxai_vs_{other}"] = r
    return out


def by_stratum(df: pd.DataFrame) -> List[dict]:
    rows = []
    for (m, level, correct), d in df.groupby(["method", "level", "correct"]):
        rows.append({"method": m, "level": level, "label_correct": bool(correct), "n": len(d),
                     "override_rate": round(float(d["override"].mean()), 4)})
    return rows


def load_judge(tag: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "cuef_v2" / f"per_case_{tag}.csv")
    df["correct"] = (df["level"] == "HIGH") == (df["mortality"] == 1)
    return df


def main() -> None:
    results = {}
    judges = {"qwen2.5:14b": "qwen25_14b", "phi4": "phi4"}
    frames = {}
    for name, tag in judges.items():
        path = DATA_DIR / "cuef_v2" / f"per_case_{tag}.csv"
        if not path.exists():
            logger.warning(f"Missing {path}; skipping {name}")
            continue
        df = load_judge(tag)
        frames[name] = df
        results[name] = {"summary": summarize(df), "tests": pairwise_tests(df),
                         "strata": by_stratum(df),
                         "label_accuracy": round(float(df[df["method"] == "empty"]["correct"].mean()), 4)}
        for m, r in results[name]["summary"].items():
            logger.info(f"[{name}] {m:10s} {r}")
    if len(frames) == 2:
        q, p = (frames[j].sort_values(["method", "hadm_id"]) for j in judges)
        cu_q = [results["qwen2.5:14b"]["summary"][m]["CU"] for m in METHODS]
        cu_p = [results["phi4"]["summary"][m]["CU"] for m in METHODS]
        results["cross_judge"] = {
            "spearman_CU_across_methods": round(float(spearmanr(cu_q, cu_p)[0]), 4),
            "kappa_override_per_case": round(float(cohen_kappa_score(q["override"], p["override"])), 4),
        }
        logger.info(f"Cross-judge: {results['cross_judge']}")
    with open(DATA_DIR / "cuef_v2" / "stats_v2.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()

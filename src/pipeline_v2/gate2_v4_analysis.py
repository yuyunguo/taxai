"""Analysis for the v4 explainer-ceiling audit (plan/prereg-v4-osf.md, section 5).

Reads the generation caches written by gate2_v4 and reports, per task,
explainer, and condition: within-stratum verdict AUROC (95% and Bonferroni 99%
CIs), paired differences, benchmarks, incremental value, and failure-mode
descriptives. Explainers whose cache is incomplete are reported as such.

Usage:
    python -m src.pipeline_v2.gate2_v4_analysis
"""

import json
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
from statsmodels.tools.sm_exceptions import PerfectSeparationError

from .config import DATA_DIR
from .gate2_v4 import IMV_CONDITIONS, LOW_ONLY, MORT_CONDITIONS, OUT
from .generator_check import select
from .timeline_measures_v3 import VERDICT

logger = logging.getLogger(__name__)
SCORE = {"too high": -1, "about right": 0, "too low": 1}
N_BOOT, SEED, N_EXPLAINERS = 1000, 42, 5


def verdict(text: Optional[str]) -> str:
    m = VERDICT.search(text or "")
    return m.group(1).lower() if m else "missing"


def boot_auroc(y: np.ndarray, s: np.ndarray, ref: Optional[np.ndarray] = None) -> Dict[str, object]:
    """AUROC of s (or paired AUROC difference s - ref) with percentile bootstrap CIs."""
    rng = np.random.default_rng(SEED)

    def stat(i: np.ndarray) -> float:
        return roc_auc_score(y[i], s[i]) - (roc_auc_score(y[i], ref[i]) if ref is not None else 0.0)

    full = np.arange(len(y))
    b = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            b.append(stat(i))
    return {"est": round(float(stat(full)), 4), "ci95": np.percentile(b, [2.5, 97.5]).round(4).tolist(),
            "ci99": np.percentile(b, [0.5, 99.5]).round(4).tolist()}


def gate2_status(a: Dict[str, object]) -> str:
    """H1 rule (prereg section 2): open / closed / inconclusive."""
    if a["est"] >= 0.60 and a["ci95"][0] > 0.5:
        return "open"
    if a["ci95"][1] < 0.60:
        return "closed"
    return "inconclusive"


def failure_modes(v: pd.Series, y: np.ndarray) -> Dict[str, object]:
    s = v.map(SCORE).fillna(0).values
    return {"distribution": v.value_counts().to_dict(), "missing_rate": round(float((v == "missing").mean()), 4),
            "about_right_share": round(float((v == "about right").mean()), 4),
            "mean_score_nonevents": round(float(s[y == 0].mean()), 4),
            "mean_score_events": round(float(s[y == 1].mean()), 4)}


def load_cache(path) -> Dict[tuple, str]:
    out: Dict[tuple, str] = {}
    if path.exists():
        for line in open(path):
            r = json.loads(line)
            out[(r["method"], int(r["hadm_id"]))] = r["response"]
    return out


def stratum_report(df: pd.DataFrame, conds: List[str], ycol: str) -> Dict[str, object]:
    y = df[ycol].values
    r: Dict[str, object] = {"n": int(len(df)), "events": int(y.sum())}
    if "pA" in df:
        r["benchmark_pA"] = boot_auroc(y, df["pA"].values)
        r["benchmark_pB"] = boot_auroc(y, df["pB"].values)
    r["rule_control"] = boot_auroc(y, df["rule_score"].values)
    scores = {}
    for c in conds:
        v = df[f"v_{c}"]
        scores[c] = v.map(SCORE).fillna(0).values
        r[c] = {"auroc": boot_auroc(y, scores[c]), **failure_modes(v, y)}
    if "static_llm" in scores and "taxai" in scores:
        r["taxai_minus_static"] = boot_auroc(y, scores["taxai"], scores["static_llm"])
    if "taxai_hint" in scores:
        r["hint_minus_taxai"] = boot_auroc(y, scores["taxai_hint"], scores["taxai"])
    if "taxai" in scores and "pA" in df:
        p = np.clip(df["pA"].values, 1e-4, 1 - 1e-4)
        X = sm.add_constant(np.column_stack([np.log(p / (1 - p)), scores["taxai"]]))
        try:
            f = sm.Logit(y, X).fit(disp=0)
            r["incremental_taxai"] = {"coef": round(float(f.params[2]), 4), "p": float(f.pvalues[2])}
        except (np.linalg.LinAlgError, PerfectSeparationError) as e:
            r["incremental_taxai"] = {"error": str(e)}
    return r


def _explainer_frame(cases: pd.DataFrame, cache: Dict[tuple, str], conds: List[str]) -> Optional[pd.DataFrame]:
    df = cases.copy()
    for c in conds:
        texts = [cache.get((c, int(h))) for h in df["hadm_id"]]
        if any(t is None for t in texts):
            return None
        df[f"v_{c}"] = pd.Series([verdict(t) for t in texts], index=df.index)
    return df


def scale_ladder(cases: pd.DataFrame) -> Dict[str, object]:
    """Secondary: paired TAXAI verdict AUROC differences up the qwen2.5 ladder, IMV LOW."""
    low = cases[cases["level"] == "LOW"]
    score: Dict[str, np.ndarray] = {}
    for model in ("qwen2.5:14b", "qwen2.5:32b", "qwen2.5:72b-instruct-q3_K_M"):
        cache = load_cache(OUT / f"gen_imv_{model.replace(':', '_').replace('.', '')}.jsonl")
        texts = [cache.get(("taxai", int(h))) for h in low["hadm_id"]]
        if all(t is not None for t in texts):
            score[model] = pd.Series([verdict(t) for t in texts]).map(SCORE).fillna(0).values
    big, y = "qwen2.5:72b-instruct-q3_K_M", low["y"].values
    if big not in score:
        return {"status": "incomplete"}
    return {f"72B_minus_{m.split(':')[1].upper()}": boot_auroc(y, score[big], score[m])
            for m in ("qwen2.5:14b", "qwen2.5:32b") if m in score}


def analyze_imv() -> Dict[str, object]:
    cases = pd.read_csv(OUT / "imv_sample.csv")
    res: Dict[str, object] = {}
    for model, conds in IMV_CONDITIONS.items():
        tag = model.replace(":", "_").replace(".", "")
        sub = cases[cases["level"] == "LOW"] if model in LOW_ONLY else cases
        df = _explainer_frame(sub, load_cache(OUT / f"gen_imv_{tag}.jsonl"), list(conds))
        if df is None:
            res[model] = "incomplete"
            continue
        mr: Dict[str, object] = {}
        for level in ("LOW", "HIGH"):
            s = df[df["level"] == level]
            if s.empty:
                continue
            mr[level] = stratum_report(s, list(conds), "y")
            for split in ("val", "test"):
                ss = s[s["split"] == split]
                if ss["y"].nunique() == 2:
                    mr[level][f"benchmarks_{split}"] = {"pA": boot_auroc(ss["y"].values, ss["pA"].values),
                                                        "pB": boot_auroc(ss["y"].values, ss["pB"].values)}
        low = df[(df["level"] == "LOW") & ~df["in_prior_sample"]]
        mr["LOW_excluding_prior_sample"] = {c: boot_auroc(low["y"].values, low[f"v_{c}"].map(SCORE).fillna(0).values)
                                            for c in conds}
        a = mr["LOW"]["taxai"]["auroc"]
        mr["gate2_status_taxai"] = gate2_status(a)
        mr["gate2_open_taxai_bonferroni"] = bool(a["est"] >= 0.60 and a["ci99"][0] > 0.5)
        res[model] = mr
    res["scale_ladder_LOW"] = scale_ladder(cases)
    low = cases[cases["level"] == "LOW"]
    rc = boot_auroc(low["y"].values, low["rule_score"].values)
    res["rule_control_LOW"] = rc
    res["H2_instrument_valid"] = bool(rc["ci95"][0] > 0.5)
    q = res.get("qwen2.5:14b")
    if isinstance(q, dict):
        res["H3_hint_helps"] = bool(q["LOW"]["hint_minus_taxai"]["ci95"][0] > 0)
    return res


def analyze_mortality() -> Dict[str, object]:
    meta = json.load(open(DATA_DIR / "v3" / "main" / "sample_meta.json"))
    cases = pd.DataFrame(select(meta["cases"]))
    ab = pd.read_csv(OUT / "mortality_ab_test.csv")
    cases = cases.merge(ab, on="hadm_id", how="left")
    cases["rule_score"] = np.where(cases["pB"] > cases["pA"], 1, -1)
    cases = cases.drop(columns=["pA", "pB"])  # benchmarks for mortality come from info_ceiling
    existing = {"qwen2.5:14b": json.load(open(DATA_DIR / "v3" / "main" / "explanations.json"))}
    caches = {"qwen2.5:14b": {(m, int(h)): t for m, d in existing["qwen2.5:14b"].items() if isinstance(d, dict)
                              for h, t in d.items()},
              "qwen2.5:32b": load_cache(DATA_DIR / "v3" / "main" / "generator_check_qwen25_32b.jsonl")}
    grid = {"qwen2.5:14b": ("static_llm", "taxai"), "qwen2.5:32b": ("static_llm", "taxai"), **MORT_CONDITIONS}
    res: Dict[str, object] = {}
    for model, conds in grid.items():
        tag = model.replace(":", "_").replace(".", "")
        cache = caches.get(model) or load_cache(OUT / f"gen_mortality_{tag}.jsonl")
        df = _explainer_frame(cases, cache, list(conds))
        if df is None:
            res[model] = "incomplete"
            continue
        res[model] = {lv: stratum_report(df[df["level"] == lv], list(conds), "mortality") for lv in ("HIGH", "LOW")}
    return res


def main() -> None:
    res = {"imv": analyze_imv(), "mortality": analyze_mortality(), "n_explainers_for_bonferroni": N_EXPLAINERS}
    json.dump(res, open(OUT / "gate2_v4.json", "w"), indent=2)
    logger.info(json.dumps({k: res["imv"].get(k) for k in ("H2_instrument_valid", "H3_hint_helps")}))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()

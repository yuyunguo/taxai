"""Pre-specified v3 statistics (plan/analysis-plan-v3.md).

Primary (H1): mean dBrier(TAXAI) - mean dBrier(Static LLM), where
dBrier = (post - y)^2 - (pre - y)^2 (negative = explanation moved the judge
toward the truth). Paired case bootstrap, 2,000 resamples, two-sided.

Secondary (Bonferroni over 8 tests):
  S1-S3  dBrier TAXAI vs Empty, Random, SHAP (paired bootstrap)
  S4     post-explanation AUROC, TAXAI vs Static LLM (paired bootstrap)
  S5     appropriate shift (moves >= 5 points toward y), TAXAI vs Static LLM (McNemar)
  S6     derived BAR (post estimate on the label's side of the threshold), TAXAI vs Static (bootstrap)
  S7     verified temporal claims per explanation, TAXAI vs Static (Wilcoxon)
  S8     mediation: dBrier ~ verified temporal claims + arm (OLS, case-clustered SEs)
Cases with an unparsed estimate in either compared arm are excluded pairwise.

Usage:
    python -m src.pipeline_v2.stats_v3 --set main --judge qwen2.5:14b
"""

import argparse
import json
import logging
from typing import Callable, Dict

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score
from statsmodels.stats.contingency_tables import mcnemar

from .config import DATA_DIR

logger = logging.getLogger(__name__)
METHODS = ["empty", "random", "shap", "static_llm", "taxai"]
N_BOOT, SEED, N_SECONDARY, SHIFT = 2000, 42, 8, 0.05


def _boot(a: pd.DataFrame, b: pd.DataFrame, stat: Callable[[pd.DataFrame], float]) -> Dict[str, object]:
    rng = np.random.default_rng(SEED)
    n = len(a)
    diffs = []
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        diffs.append(stat(a.iloc[i]) - stat(b.iloc[i]))
    diffs = np.array(diffs)
    p = max(2 * min((diffs <= 0).mean(), (diffs >= 0).mean()), 1 / N_BOOT)
    return {"diff": round(stat(a) - stat(b), 5), "ci95": np.percentile(diffs, [2.5, 97.5]).round(5).tolist(),
            "p": float(p)}


def _bar(d: pd.DataFrame, thr: float) -> float:
    accept = (d["post"] >= thr) == (d["level"] == "HIGH")
    correct = (d["level"] == "HIGH") == (d["mortality"] == 1)
    return 0.5 * (accept[correct].mean() + (~accept[~correct]).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["pilot", "main"], required=True)
    ap.add_argument("--judge", required=True)
    args = ap.parse_args()
    od = DATA_DIR / "v3" / args.set
    tag = args.judge.replace(":", "_").replace(".", "")
    thr = json.load(open(od / "sample_meta.json"))["threshold"]
    df = pd.read_csv(od / f"per_case_{tag}.csv")
    df["y"] = df["mortality"].astype(float)
    df["dbrier"] = (df["post"] - df["y"]) ** 2 - (df["pre"] - df["y"]) ** 2
    df["toward"] = ((df["post"] - df["pre"]) * np.where(df["y"] == 1, 1, -1)) >= SHIFT
    faith = pd.read_csv(od / "faithfulness_v3_per_case.csv")
    df = df.merge(faith[["hadm_id", "method", "n_temporal_supported"]], on=["hadm_id", "method"], how="left")
    df["n_temporal_supported"] = df["n_temporal_supported"].fillna(0)

    res: Dict[str, object] = {"n_cases": int(df["hadm_id"].nunique()),
                              "parse_rate": round(float(df[["pre_parsed", "post_parsed"]].values.mean()), 4)}
    pre = df.drop_duplicates("hadm_id").dropna(subset=["pre"])
    res["pre"] = {"brier": round(float(((pre["pre"] - pre["y"]) ** 2).mean()), 4),
                  "auroc": round(float(roc_auc_score(pre["y"], pre["pre"])), 4)}
    res["by_method"] = {}
    for m in METHODS:
        d = df[df["method"] == m].dropna(subset=["pre", "post"])
        res["by_method"][m] = {"n": len(d), "dBrier": round(float(d["dbrier"].mean()), 5),
                               "post_brier": round(float(((d["post"] - d["y"]) ** 2).mean()), 4),
                               "post_auroc": round(float(roc_auc_score(d["y"], d["post"])), 4),
                               "mean_abs_shift": round(float((d["post"] - d["pre"]).abs().mean()), 4),
                               "toward_rate": round(float(d["toward"].mean()), 4),
                               "BAR_derived": round(float(_bar(d, thr)), 4)}

    def pair(m1: str, m2: str):
        a = df[df["method"] == m1].set_index("hadm_id")
        b = df[df["method"] == m2].set_index("hadm_id")
        ok = a[["pre", "post"]].notna().all(1) & b[["pre", "post"]].notna().all(1)
        return a[ok].sort_index(), b[ok].sort_index()

    t, s = pair("taxai", "static_llm")
    res["H1_primary"] = {"n_pairs": len(t), **_boot(t, s, lambda d: d["dbrier"].mean())}
    sec: Dict[str, Dict[str, object]] = {}
    for k, other in (("S1", "empty"), ("S2", "random"), ("S3", "shap")):
        a, b = pair("taxai", other)
        sec[f"{k}_dBrier_vs_{other}"] = _boot(a, b, lambda d: d["dbrier"].mean())
    sec["S4_post_auroc_vs_static"] = _boot(t, s, lambda d: roc_auc_score(d["y"], d["post"]))
    b_, c_ = int((t["toward"] & ~s["toward"]).sum()), int((~t["toward"] & s["toward"]).sum())
    sec["S5_toward_vs_static"] = {"diff": round(float(t["toward"].mean() - s["toward"].mean()), 4),
                                  "p": float(mcnemar([[0, b_], [c_, 0]], exact=False, correction=True).pvalue) if b_ + c_ else 1.0}
    sec["S6_BAR_vs_static"] = _boot(t, s, lambda d: _bar(d.reset_index(), thr))
    w = wilcoxon(t["n_temporal_supported"], s["n_temporal_supported"]) if (t["n_temporal_supported"] != s["n_temporal_supported"]).any() else None
    sec["S7_temporal_claims_vs_static"] = {"taxai_mean": round(float(t["n_temporal_supported"].mean()), 3),
                                           "static_mean": round(float(s["n_temporal_supported"].mean()), 3),
                                           "p": float(w.pvalue) if w else 1.0}
    med = pd.concat([t.assign(arm="taxai"), s.assign(arm="static_llm")]).reset_index()
    fit = smf.ols("dbrier ~ n_temporal_supported + C(arm)", data=med).fit(
        cov_type="cluster", cov_kwds={"groups": med["hadm_id"]})
    sec["S8_mediation_temporal_claims"] = {"coef": round(float(fit.params["n_temporal_supported"]), 6),
                                           "ci95": fit.conf_int().loc["n_temporal_supported"].round(6).tolist(),
                                           "p": float(fit.pvalues["n_temporal_supported"])}
    for v in sec.values():
        v["p_bonf"] = min(1.0, v["p"] * N_SECONDARY)
    res["secondary"] = sec
    json.dump(res, open(od / f"stats_v3_{tag}.json", "w"), indent=2)
    logger.info(json.dumps(res, indent=1))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()

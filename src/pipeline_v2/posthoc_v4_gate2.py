"""Post hoc Gate 2 re-analyses after the simulated review (plan/posthoc-v4-tierA.md, P1-P4).

Uses only the registered v4 sample and generation caches; no new generation.
Labelled post hoc; registered classifications are unchanged.

Usage:
    python -m src.pipeline_v2.posthoc_v4_gate2 [--exclude-ids FILE]
"""

import argparse
import json
import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import chi2, spearmanr
from sklearn.metrics import roc_auc_score

from .gate2_v4 import OUT
from .gate2_v4_analysis import SCORE, load_cache, verdict

logger = logging.getLogger(__name__)
N_BOOT, SEED = 1000, 42
EXPLAINERS = ("qwen2.5:14b", "qwen2.5:32b", "qwen2.5:72b-instruct-q3_K_M", "gemma3:27b", "gpt-oss:20b")


def boot(y: np.ndarray, f, seed: int = SEED) -> Dict[str, object]:
    """Percentile bootstrap of a statistic f(index array)."""
    rng = np.random.default_rng(seed)
    full = f(np.arange(len(y)))
    b = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            b.append(f(i))
    return {"est": round(float(full), 4), "ci95": np.percentile(b, [2.5, 97.5]).round(4).tolist()}


def max_auroc(scores: np.ndarray, y: np.ndarray) -> float:
    """Highest AUROC attainable with the observed verdict counts: highest scores assigned to events first."""
    s = np.sort(scores)[::-1]
    yy = np.zeros(len(s), dtype=int)
    yy[: int(y.sum())] = 1
    return float(roc_auc_score(yy, s)) if 0 < yy.sum() < len(yy) else float("nan")


def firth(X: np.ndarray, y: np.ndarray, iters: int = 100) -> Dict[str, np.ndarray]:
    """Firth-penalized logistic regression (Jeffreys prior); returns coefficients, SEs, penalized log-likelihood."""
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ beta))
        w = p * (1 - p)
        info = X.T @ (X * w[:, None])
        inv = np.linalg.inv(info)
        h = np.einsum("ij,jk,ik->i", X * np.sqrt(w)[:, None], inv, X * np.sqrt(w)[:, None])
        step = inv @ (X.T @ (y - p + h * (0.5 - p)))
        beta = beta + step
        if np.max(np.abs(step)) < 1e-8:
            break
    p = 1 / (1 + np.exp(-X @ beta))
    w = p * (1 - p)
    info = X.T @ (X * w[:, None])
    ll = float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)) + 0.5 * np.linalg.slogdet(info)[1])
    return {"beta": beta, "se": np.sqrt(np.diag(np.linalg.inv(info))), "ll": ll}


def incremental(y: np.ndarray, pa: np.ndarray, v: pd.Series, min_ref: int = 5) -> Dict[str, object]:
    """Firth model of outcome on logit(pA) plus verdict indicators.

    Reference category is "about right" (missing verdicts pooled with it). If it holds fewer than
    `min_ref` cases, the most frequent remaining category is used instead (deviation logged in
    plan/posthoc-v4-tierA-deviations.md); the "too low" coefficient keeps its direction.
    """
    lp = np.log(np.clip(pa, 1e-4, 1 - 1e-4) / (1 - np.clip(pa, 1e-4, 1 - 1e-4)))
    cat = v.where(v.isin(["too low", "too high"]), "about right")
    ref = "about right"
    if (cat == ref).sum() < min_ref:
        ref = cat[cat != "about right"].value_counts().idxmax()
    sparse = [c for c in ("too low", "too high", "about right") if c != ref and 0 < (cat == c).sum() < min_ref]
    cat = cat.where(~cat.isin(sparse), ref)  # pool sparse categories with the reference
    levels = [c for c in ("too low", "too high", "about right") if c != ref and (cat == c).sum() > 0]
    dummies = [(cat == c).astype(float).values for c in levels]
    out: Dict[str, object] = {"reference": ref, "pooled_into_reference": sparse}
    try:
        base = firth(np.column_stack([np.ones(len(y)), lp]), y)
        full = firth(np.column_stack([np.ones(len(y)), lp] + dummies), y)
    except np.linalg.LinAlgError as e:
        out["error"] = f"fit failed: {e}"
        out["added_information"] = False
        return out
    df = len(dummies)
    lr = 2 * (full["ll"] - base["ll"])
    out.update({"lr_stat": round(lr, 3), "lr_df": df, "lr_p": round(float(chi2.sf(lr, df)), 4) if df else None})
    for k, name in enumerate(levels):
        b, se = full["beta"][2 + k], full["se"][2 + k]
        out[name.replace(" ", "_")] = {"coef": round(float(b), 3),
                                       "ci95": [round(float(b - 1.96 * se), 3), round(float(b + 1.96 * se), 3)]}
    tl = out.get("too_low")
    if ref == "too low":  # direction must come from the other categories relative to "too low"
        tl = None
    out["added_information"] = bool(out["lr_p"] is not None and out["lr_p"] < 0.05 and tl is not None and tl["ci95"][0] > 0)
    return out


def stratified_auroc(y: np.ndarray, s: np.ndarray, pa: np.ndarray) -> Optional[float]:
    bins = pd.qcut(pa, 3, labels=False, duplicates="drop")
    num, den = 0.0, 0.0
    for b in np.unique(bins):
        m = bins == b
        ne, nn = int(y[m].sum()), int((1 - y[m]).sum())
        if ne and nn:
            num += roc_auc_score(y[m], s[m]) * ne * nn
            den += ne * nn
    return round(num / den, 4) if den else None


def run(exclude: Optional[set] = None) -> Dict[str, object]:
    cases = pd.read_csv(OUT / "imv_sample.csv")
    low = cases[cases["level"] == "LOW"].copy()
    if exclude:
        low = low[~low["hadm_id"].isin(exclude)]
    y, pa, pb = low["y"].values, low["pA"].values, low["pB"].values
    rule = low["rule_score"].values
    res: Dict[str, object] = {"n": int(len(low)), "events": int(y.sum())}
    res["rule_control"] = boot(y, lambda i: roc_auc_score(y[i], rule[i]))
    res["B_minus_A"] = boot(y, lambda i: roc_auc_score(y[i], pb[i]) - roc_auc_score(y[i], pa[i]))
    by_split: Dict[str, object] = {}
    for sp in ("val", "test"):
        m = (low["split"] == sp).values
        ys, pas, pbs, rs = y[m], pa[m], pb[m], rule[m]
        by_split[sp] = {"n": int(m.sum()), "events": int(ys.sum()),
                        "pA": round(float(roc_auc_score(ys, pas)), 4), "pB": round(float(roc_auc_score(ys, pbs)), 4),
                        "B_minus_A": boot(ys, lambda i: roc_auc_score(ys[i], pbs[i]) - roc_auc_score(ys[i], pas[i])),
                        "rule_control": round(float(roc_auc_score(ys, rs)), 4)}
    res["by_split"] = by_split
    explainers: Dict[str, object] = {}
    for model in EXPLAINERS:
        cache = load_cache(OUT / f"gen_imv_{model.replace(':', '_').replace('.', '')}.jsonl")
        conds = ("static_llm", "taxai", "taxai_hint") if model == "qwen2.5:14b" else ("static_llm", "taxai")
        mr: Dict[str, object] = {}
        for c in conds:
            texts = [cache.get((c, int(h))) for h in low["hadm_id"]]
            if any(t is None for t in texts):
                continue
            v = pd.Series([verdict(t) for t in texts], index=low.index)
            s = v.map(SCORE).fillna(0).values
            top = v.value_counts(normalize=True).max()
            r: Dict[str, object] = {
                "auroc": round(float(roc_auc_score(y, s)), 4),
                "max_attainable_auroc": round(max_auroc(s, y), 4),
                "dissent_rate_events": round(float((v[y == 1] != "about right").mean()), 4),
                "dissent_rate_nonevents": round(float((v[y == 0] != "about right").mean()), 4),
                "non_responsive": bool(top > 0.90),
                "minus_rule_control": boot(y, lambda i: roc_auc_score(y[i], s[i]) - roc_auc_score(y[i], rule[i])),
                "spearman_with_pA": round(float(spearmanr(s, pa).correlation), 4) if np.std(s) > 0 else None,
                "stratified_auroc_pA_tertiles": stratified_auroc(y, s, pa),
                "incremental_firth": incremental(y, pa, v),
            }
            if c == "taxai":
                r["by_split"] = {sp: round(float(roc_auc_score(y[(low['split'] == sp).values],
                                                                s[(low['split'] == sp).values])), 4)
                                 for sp in ("val", "test")}
            ci = r["minus_rule_control"]["ci95"]
            r["below_positive_control"] = bool(ci[1] < 0)
            mr[c] = r
        explainers[model] = mr
    res["explainers"] = explainers
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exclude-ids", help="file with one hadm_id per line to exclude (P7d sensitivity)")
    ap.add_argument("--out", default="posthoc_v4_gate2.json")
    args = ap.parse_args()
    exclude = set(int(x) for x in open(args.exclude_ids).read().split()) if args.exclude_ids else None
    res = run(exclude)
    json.dump(res, open(OUT / args.out, "w"), indent=2)
    logger.info(f"wrote {OUT / args.out}; n={res['n']}, events={res['events']}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()

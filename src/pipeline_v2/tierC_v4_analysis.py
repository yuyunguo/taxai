"""Tier C analysis (plan/prereg-v4-addendum-tierC.md, sections 3-4).

Usage:
    python -m src.pipeline_v2.tierC_v4_analysis
"""

import json
import logging
from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import chi2, spearmanr
from sklearn.metrics import roc_auc_score

from .gate2_v4 import OUT
from .gate2_v4_analysis import SCORE, gate2_status, load_cache, verdict
from .posthoc_v4_gate2 import firth, max_auroc
from .prompts_tierC import parse_elicited
from .tierC_v4 import GRID

logger = logging.getLogger(__name__)
N_BOOT, SEED = 1000, 42


def boot(y: np.ndarray, f) -> Dict[str, object]:
    rng = np.random.default_rng(SEED)
    b = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            b.append(f(i))
    return {"est": round(float(f(np.arange(len(y)))), 4), "ci95": np.percentile(b, [2.5, 97.5]).round(4).tolist(),
            "ci99": np.percentile(b, [0.5, 99.5]).round(4).tolist()}


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def elicited_endpoints(y: np.ndarray, pa: np.ndarray, pe: np.ndarray) -> Dict[str, object]:
    base = firth(np.column_stack([np.ones(len(y)), logit(pa)]), y)
    full = firth(np.column_stack([np.ones(len(y)), logit(pa), logit(pe)]), y)
    b, se = full["beta"][2], full["se"][2]
    lr = 2 * (full["ll"] - base["ll"])
    lr_p = float(chi2.sf(lr, 1))
    r = {"auroc": boot(y, lambda i: roc_auc_score(y[i], pe[i])),
         "minus_modelA": boot(y, lambda i: roc_auc_score(y[i], pe[i]) - roc_auc_score(y[i], pa[i])),
         "firth_coef": round(float(b), 3), "firth_ci99": [round(float(b - 2.576 * se), 3), round(float(b + 2.576 * se), 3)],
         "lr_p": round(lr_p, 4), "spearman_with_pA": round(float(spearmanr(pe, pa).correlation), 4)}
    r["conveys_beyond_prediction"] = bool(lr_p < 0.01 and r["firth_ci99"][0] > 0)
    return r


def main() -> None:
    cases = pd.read_csv(OUT / "imv_sample.csv")
    low = cases[cases["level"] == "LOW"].reset_index(drop=True)
    y, pa = low["y"].values, low["pA"].values
    res: Dict[str, object] = {"n": int(len(low)), "events": int(y.sum())}
    for model, conds in GRID.items():
        tag = model.replace(":", "_").replace(".", "")
        cache = load_cache(OUT / f"gen_tierC_{tag}.jsonl")
        reg = load_cache(OUT / f"gen_imv_{tag}.jsonl")
        reg_s = np.array([SCORE.get(verdict(reg.get(("taxai", int(h)))), 0) for h in low["hadm_id"]], dtype=float)
        mr: Dict[str, object] = {}
        for cond in conds:
            texts = [cache.get((cond, int(h))) for h in low["hadm_id"]]
            if any(t is None for t in texts):
                mr[cond] = "incomplete"
                continue
            if cond in ("C1a", "C1b"):
                pe_raw = np.array([parse_elicited(t) if parse_elicited(t) is not None else np.nan for t in texts])
                ok = ~np.isnan(pe_raw)
                r: Dict[str, object] = {"parse_rate": round(float(ok.mean()), 4),
                                        "median_elicited": round(float(np.nanmedian(pe_raw)), 4) if ok.any() else None}
                if ok.sum() > 20 and len(np.unique(y[ok])) == 2 and np.nanstd(pe_raw) > 0:
                    r["parsed_cases"] = elicited_endpoints(y[ok], pa[ok], pe_raw[ok])
                    imputed = np.where(ok, pe_raw, pa)
                    r["imputed_with_pA"] = elicited_endpoints(y, pa, imputed)
                    if cond == "C1b":
                        r["discriminates_without_prediction"] = bool(r["parsed_cases"]["auroc"]["ci99"][0] > 0.5)
                else:
                    r["note"] = "insufficient variation or parses for analysis"
                mr[cond] = r
            else:
                v = pd.Series([verdict(t) for t in texts])
                s = v.map(SCORE).fillna(0).values
                a = boot(y, lambda i: roc_auc_score(y[i], s[i]))
                diff = boot(y, lambda i: roc_auc_score(y[i], s[i]) - roc_auc_score(y[i], reg_s[i]))
                r = {"auroc": a, "minus_registered_taxai": diff, "max_attainable_auroc": round(max_auroc(s, y), 4),
                     "distribution": v.value_counts().to_dict(),
                     "about_right_share": round(float((v == "about right").mean()), 4),
                     "registered_about_right_share": round(float((reg_s == 0).mean()), 4)}
                if cond == "C2":
                    r["hint_supported"] = bool(diff["ci99"][0] > 0)
                if cond == "C3":
                    r["status"] = gate2_status(a)
                    r["instruction_dependent"] = bool(r["status"] != "closed" or diff["ci99"][0] > 0 or diff["ci99"][1] < 0)
                mr[cond] = r
        res[model] = mr
    json.dump(res, open(OUT / "tierC_v4.json", "w"), indent=2)
    logger.info(f"wrote {OUT / 'tierC_v4.json'}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()

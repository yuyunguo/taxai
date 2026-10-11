"""Generator-capability check (plan/generator-check.md; post hoc, specified before running).

Regenerates v3 explanations with a larger generator for an error-enriched
subset and compares within-label verdict discrimination with qwen2.5:14b.

Usage:
    python -m src.pipeline_v2.generator_check --model qwen2.5:32b [--limit N]
"""

import argparse
import json
import logging
import pickle
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .config import DATA_DIR
from .llm import CachedLLM
from .prompts import prediction_line
from .prompts_v3 import EXPLAIN_V3, render_chart, render_summary
from .timeline_measures_v3 import VERDICT, right_verdict

logger = logging.getLogger(__name__)
SCORE = {"too high": -1, "about right": 0, "too low": 1}


def select(cases: list) -> list:
    df = pd.DataFrame(cases)
    df["wrong"] = (df["level"] == "HIGH") != (df["mortality"] == 1)
    rng = np.random.default_rng(11)
    out = []
    for level in ("HIGH", "LOW"):
        s = df[df["level"] == level]
        wrong, right = s[s["wrong"]], s[~s["wrong"]]
        pick = rng.choice(right.index, size=len(wrong), replace=False)
        out += wrong.to_dict("records") + right.loc[pick].to_dict("records")
    return out


def verdict(text: str) -> str:
    m = VERDICT.search(text or "")
    return m.group(1).lower() if m else "missing"


def auroc_ci(y: np.ndarray, s: np.ndarray, seed: int = 42) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    n, boots = len(y), []
    for _ in range(1000):
        i = rng.integers(0, n, n)
        if y[i].min() != y[i].max():
            boots.append(roc_auc_score(y[i], s[i]))
    return {"auroc": round(float(roc_auc_score(y, s)), 4), "ci95": np.percentile(boots, [2.5, 97.5]).round(4).tolist()}


def paired_diff(y: np.ndarray, a: np.ndarray, b: np.ndarray, seed: int = 42) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    n, d = len(y), []
    for _ in range(1000):
        i = rng.integers(0, n, n)
        if y[i].min() != y[i].max():
            d.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    return {"diff": round(float(roc_auc_score(y, a) - roc_auc_score(y, b)), 4),
            "ci95": np.percentile(d, [2.5, 97.5]).round(4).tolist()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    od = DATA_DIR / "v3" / "main"
    meta = json.load(open(od / "sample_meta.json"))
    thr = meta["threshold"]
    cases = select(meta["cases"])
    cases = cases[: args.limit] if args.limit else cases
    with open(DATA_DIR / "trajectories_test_v3.pkl", "rb") as f:
        trajs = pickle.load(f)
    base = json.load(open(od / "explanations.json"))
    tag = args.model.replace(":", "_").replace(".", "")
    llm = CachedLLM(od / f"generator_check_{tag}.jsonl")
    rows = []
    for i, c in enumerate(cases):
        h, t = c["hadm_id"], trajs[c["hadm_id"]]
        pl = prediction_line(c["pred_proba"], c["level"], thr)
        summary = f"TRAJECTORY SUMMARY:\n{render_summary(t)}"
        data = {"static_llm": summary, "taxai": f"{summary}\n\n{render_chart(t)}"}
        for m in ("static_llm", "taxai"):
            text = llm.ask(args.model, "explain_v3", m, h, EXPLAIN_V3.format(prediction=pl, data=data[m], level=c["level"]))
            rows.append({"hadm_id": h, "level": c["level"], "mortality": c["mortality"], "method": m,
                         "verdict_big": verdict(text), "verdict_14b": verdict(base[m][str(h)]),
                         "right": right_verdict(c["level"], c["mortality"])})
        if (i + 1) % 25 == 0:
            logger.info(f"  {i + 1}/{len(cases)}")
    df = pd.DataFrame(rows)
    df.to_csv(od / f"generator_check_{tag}.csv", index=False)
    res: Dict[str, object] = {"model": args.model, "n_cases": len(cases)}
    for m, g in df.groupby("method"):
        r = {"verdicts_big": g["verdict_big"].value_counts().to_dict(),
             "verdicts_14b": g["verdict_14b"].value_counts().to_dict(),
             "accuracy_big": round(float((g["verdict_big"] == g["right"]).mean()), 4),
             "accuracy_14b": round(float((g["verdict_14b"] == g["right"]).mean()), 4)}
        for level in ("HIGH", "LOW"):
            s = g[g["level"] == level]
            y = s["mortality"].values
            sb, s14 = s["verdict_big"].map(SCORE).fillna(0).values, s["verdict_14b"].map(SCORE).fillna(0).values
            r[f"{level}_big"] = auroc_ci(y, sb)
            r[f"{level}_14b"] = auroc_ci(y, s14)
            r[f"{level}_diff_big_minus_14b"] = paired_diff(y, sb, s14)
        res[m] = r
    h = res.get("taxai", {})
    res["decision_capability_bottleneck_taxai"] = bool(
        h and h["HIGH_big"]["auroc"] >= 0.60 and h["HIGH_big"]["ci95"][0] > 0.5 and h["HIGH_diff_big_minus_14b"]["ci95"][0] > 0)
    json.dump(res, open(od / f"generator_check_{tag}.json", "w"), indent=2)
    logger.info(json.dumps(res, indent=1))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()

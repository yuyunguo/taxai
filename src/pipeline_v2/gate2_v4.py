"""Explainer-ceiling audit, v4 (plan/prereg-v4-osf.md; pre-registered).

Phase 1 (--build): trains models A (aggregates) and B (+ dynamics) for 24-h
invasive ventilation (IMV) and for mortality, draws the IMV Gate 2 sample from
validation + test, and writes the sample and the rule-control scores.
Phase 2 (--explainer MODEL --task imv|mortality): generates explanations for
the pre-registered conditions into a per-model JSONL cache.

Usage (on the database host):
    PGPASSWORD=... python -m src.pipeline_v2.gate2_v4 --build [--dry-run]
    python -m src.pipeline_v2.gate2_v4 --explainer gemma3:27b --task imv [--limit N]
"""

import argparse
import json
import logging
import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from .build import load_events
from .config import DATA_DIR
from .gate1_outcomes import outcome_labels
from .gate2_imv import EXPLAIN_IMV, prediction_line_imv
from .generator_check import select
from .info_ceiling import dynamics_features, fit
from .llm import CachedLLM
from .prompts import prediction_line
from .prompts_v3 import EXPLAIN_V3, render_chart, render_summary

logger = logging.getLogger(__name__)
OUT = DATA_DIR / "v4"
MAX_EVENTS = 200
SAMPLE_SEED = 11
HINT = "A second model that also accounts for the timing of recent changes estimates this risk at {p:.1%}."

# Pre-registered explainer x condition grid (prereg section 4).
IMV_CONDITIONS: Dict[str, Tuple[str, ...]] = {
    "qwen2.5:14b": ("static_llm", "taxai", "taxai_hint"),
    "qwen2.5:32b": ("static_llm", "taxai"),
    "gemma3:27b": ("static_llm", "taxai"),
    "gpt-oss:20b": ("static_llm", "taxai"),
    "qwen2.5:72b-instruct-q3_K_M": ("taxai",),
}
LOW_ONLY = ("qwen2.5:72b-instruct-q3_K_M",)  # ~75 s per explanation; IMV HIGH stratum skipped (prereg section 4)
# Explainers served by another Ollama host (env var, never hard-coded); others use LLM.host.
# gpt-oss reasoning tokens share the context window; longest TAXAI prompt is ~6.6k tokens (prereg section 4).
OPTION_OVERRIDES: Dict[str, Dict[str, float]] = {"gpt-oss:20b": {"num_ctx": 16384}}
REMOTE_HOST_ENV = {"qwen2.5:72b-instruct-q3_K_M": "MAC_OLLAMA_HOST"}
MORT_CONDITIONS: Dict[str, Tuple[str, ...]] = {
    "gemma3:27b": ("static_llm", "taxai"),
    "gpt-oss:20b": ("static_llm", "taxai"),
    "qwen2.5:72b-instruct-q3_K_M": ("taxai",),
}


def _threshold(y: np.ndarray, p: np.ndarray) -> float:
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-9, None)
    return float(thr[int(np.argmax(f1))])


def _fit_ab(d: pd.DataFrame, base: List[str], dyn: List[str]) -> Dict[str, Dict[str, np.ndarray]]:
    parts = {s: d[d["split"] == s] for s in ("train", "val", "test")}
    y = {s: parts[s]["y"].values for s in parts}
    prob = {}
    for key, cols in (("A", base), ("B", base + dyn)):
        X = {s: parts[s][cols].values.astype(np.float32) for s in parts}
        m = fit(X, y)
        prob[key] = {s: m.predict_proba(X[s])[:, 1] for s in ("val", "test")}
    return prob


def build(dry_run: bool) -> None:
    cohort = pd.read_parquet(DATA_DIR / "cohort.parquet")
    feat = pd.read_parquet(DATA_DIR / "features.parquet").set_index("hadm_id")
    dyn = dynamics_features(load_events(cohort))
    full = feat.join(dyn)
    base = [c for c in feat.columns if c not in ("mortality", "split")]
    dyn_cols = list(dyn.columns)

    lab = outcome_labels(cohort)["invasive_ventilation"]
    d = full.loc[lab["hadm_id"]].copy()
    d["y"] = lab["event"].values
    prob = _fit_ab(d, base, dyn_cols)
    val = d[d["split"] == "val"]
    thr = _threshold(val["y"].values, prob["A"]["val"])
    frames = []
    for s in ("val", "test"):
        part = d[d["split"] == s][["y", "split"]]
        frames.append(part.assign(pA=prob["A"][s], pB=prob["B"][s]))
    pool = pd.concat(frames)
    pool["level"] = np.where(pool["pA"] >= thr, "HIGH", "LOW")
    rng = np.random.default_rng(SAMPLE_SEED)
    picks: List[int] = []
    for level in ("HIGH", "LOW"):
        s = pool[pool["level"] == level]
        ev, ne = s[s["y"] == 1], s[s["y"] == 0]
        k = min(len(ev), MAX_EVENTS, len(ne))
        picks += list(rng.choice(ev.index, k, replace=False)) + list(rng.choice(ne.index, k, replace=False))
        logger.info(f"IMV {level}: pool events {len(ev)}, non-events {len(ne)} -> sampled {k} + {k}")
    sample = pool.loc[picks].copy()
    old = pd.read_csv(DATA_DIR / "v3" / "gate2_imv" / "gate2_imv_cases.csv")["hadm_id"].unique()
    sample["in_prior_sample"] = sample.index.isin(old)
    sample["rule_score"] = np.where(sample["pB"] > sample["pA"], 1, -1)
    logger.info(f"IMV threshold {thr:.4f}; sample {len(sample)}; overlap with prior sample "
                f"{int(sample['in_prior_sample'].sum())}; by split {sample['split'].value_counts().to_dict()}")
    if dry_run:
        return

    # Mortality: rule-control scores for the existing generator-check sample.
    dm = full.assign(y=full["mortality"])
    pm = _fit_ab(dm, base, dyn_cols)
    test_ids = dm[dm["split"] == "test"].index
    mort = pd.DataFrame({"pA": pm["A"]["test"], "pB": pm["B"]["test"]}, index=test_ids)

    OUT.mkdir(parents=True, exist_ok=True)
    sample.rename_axis("hadm_id").reset_index().to_csv(OUT / "imv_sample.csv", index=False)
    mort.rename_axis("hadm_id").reset_index().to_csv(OUT / "mortality_ab_test.csv", index=False)
    json.dump({"imv_threshold": thr, "max_events": MAX_EVENTS, "seed": SAMPLE_SEED},
              open(OUT / "build_meta.json", "w"), indent=2)


def _trajectories() -> Dict[int, dict]:
    out: Dict[int, dict] = {}
    for s in ("test", "val"):
        with open(DATA_DIR / f"trajectories_{s}_v3.pkl", "rb") as f:
            out.update(pickle.load(f))
    return out


def _inputs(t: dict) -> Dict[str, str]:
    summary = f"TRAJECTORY SUMMARY:\n{render_summary(t)}"
    return {"static_llm": summary, "taxai": f"{summary}\n\n{render_chart(t)}"}


def generate(model: str, task: str, limit: int) -> None:
    grid = IMV_CONDITIONS if task == "imv" else MORT_CONDITIONS
    if model not in grid:
        raise ValueError(f"{model} is not pre-registered for task {task}")
    trajs = _trajectories()
    tag = model.replace(":", "_").replace(".", "")
    host = os.environ[REMOTE_HOST_ENV[model]] if model in REMOTE_HOST_ENV else None
    llm = CachedLLM(OUT / f"gen_{task}_{tag}.jsonl", host=host, options=OPTION_OVERRIDES.get(model))
    info = llm.client.show(model)
    info = info.model_dump() if hasattr(info, "model_dump") else dict(info)
    info.pop("modelfile", None)
    json.dump({"model": model, "options": llm.options, "host_env": REMOTE_HOST_ENV.get(model), "show": info},
              open(OUT / f"model_info_{tag}.json", "w"), indent=2, default=str)
    if task == "imv":
        thr = json.load(open(OUT / "build_meta.json"))["imv_threshold"]
        cases = pd.read_csv(OUT / "imv_sample.csv").to_dict("records")
    else:
        meta = json.load(open(DATA_DIR / "v3" / "main" / "sample_meta.json"))
        thr, cases = meta["threshold"], select(meta["cases"])
    if task == "imv" and model in LOW_ONLY:
        cases = [c for c in cases if c["level"] == "LOW"]
    cases = cases[:limit] if limit else cases
    for i, c in enumerate(cases):
        h = int(c["hadm_id"])
        data = _inputs(trajs[h])
        for cond in grid[model]:
            if task == "imv":
                pl = prediction_line_imv(c["pA"], c["level"], thr)
                if cond == "taxai_hint":
                    pl = f"{pl}\n{HINT.format(p=c['pB'])}"
                prompt = EXPLAIN_IMV.format(prediction=pl, data=data[cond.replace("_hint", "")], level=c["level"])
            else:
                pl = prediction_line(c["pred_proba"], c["level"], thr)
                prompt = EXPLAIN_V3.format(prediction=pl, data=data[cond], level=c["level"])
            llm.ask(model, f"explain_v4_{task}", cond, h, prompt)
        if (i + 1) % 25 == 0:
            logger.info(f"[{model} {task}] {i + 1}/{len(cases)}")
    logger.info(f"[{model} {task}] done: {len(cases)} cases")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--explainer")
    ap.add_argument("--task", choices=("imv", "mortality"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    if args.build:
        build(args.dry_run)
    elif args.explainer and args.task:
        generate(args.explainer, args.task, args.limit)
    else:
        ap.error("use --build or --explainer with --task")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()

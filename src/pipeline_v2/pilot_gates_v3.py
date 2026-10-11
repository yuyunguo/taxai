"""Evaluate the pre-specified pilot go/no-go criteria (plan/study-v3-design.md).

G1  input contrast: data points in the TAXAI chart not in the summary, median >= 30
G2  judge parse rate >= 95% of turns
G3  judge pre-estimates informative: SD >= 0.10 (AUROC reported)
G4  longest judge prompt <= 6,000 tokens (estimated as characters / 4)

Usage:
    python -m src.pipeline_v2.pilot_gates_v3 --judge qwen2.5:14b
"""

import argparse
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .config import DATA_DIR


def input_contrast(traj: dict) -> int:
    """Lab values beyond first/last per test plus vital points beyond min/max per variable."""
    extra = 0
    for kind in ("lab", "vital"):
        counts = pd.Series([e["label"] for e in traj["events"] if e["type"] == kind]).value_counts()
        extra += int((counts - 2).clip(lower=0).sum()) if len(counts) else 0
    return extra


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", required=True)
    args = ap.parse_args()
    od = DATA_DIR / "v3" / "pilot"
    meta = json.load(open(od / "sample_meta.json"))["cases"]
    with open(DATA_DIR / "trajectories_test_v3.pkl", "rb") as f:
        trajs = pickle.load(f)
    tag = args.judge.replace(":", "_").replace(".", "")
    df = pd.read_csv(od / f"per_case_{tag}.csv")

    contrast = [input_contrast(trajs[c["hadm_id"]]) for c in meta]
    pre = df.drop_duplicates("hadm_id")
    parse_rate = float(np.mean(list(pre["pre_parsed"]) + list(df["post_parsed"])))
    pre_valid = pre.dropna(subset=["pre"])
    sd = float(pre_valid["pre"].std())
    auroc = float(roc_auc_score(pre_valid["mortality"], pre_valid["pre"])) if pre_valid["mortality"].nunique() > 1 else None
    max_tokens = int(df["prompt_chars"].max() / 4)
    expl = json.load(open(od / "explanations.json"))
    verdicts = {}
    for m in ("static_llm", "taxai"):
        v = pd.Series(expl[m].values()).str.extract(r"appears (too high|about right|too low)")[0].fillna("missing")
        verdicts[m] = v.value_counts().to_dict()
    gates = {
        "G1_input_contrast_median": {"value": float(np.median(contrast)), "pass": bool(np.median(contrast) >= 30)},
        "G2_parse_rate": {"value": round(parse_rate, 4), "pass": parse_rate >= 0.95},
        "G3_pre_sd": {"value": round(sd, 4), "pass": sd >= 0.10, "pre_auroc_reported": auroc},
        "G4_max_prompt_tokens_est": {"value": max_tokens, "pass": max_tokens <= 6000},
        "descriptive": {
            "n_cases": len(meta),
            "mean_abs_shift_by_method": df.assign(s=(df["post"] - df["pre"]).abs()).groupby("method")["s"].mean().round(4).to_dict(),
            "generator_verdicts": verdicts,
        },
    }
    print(json.dumps(gates, indent=2))
    json.dump(gates, open(od / f"pilot_gates_{tag}.json", "w"), indent=2)


if __name__ == "__main__":
    main()

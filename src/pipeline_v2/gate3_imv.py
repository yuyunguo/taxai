"""Limited Gate 3 test on invasive ventilation (plan/gate3-imv.md; exploratory).

Two-turn judge protocol (pre/post estimate) on the Gate 2 sample with five
arms: empty, static_llm, taxai (existing explanations), model_b and
trajectory_direction (informative positive controls).

Usage:
    python -m src.pipeline_v2.gate3_imv
"""

import json
import logging
import pickle
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .config import DATA_DIR, LLM
from .gate2_imv import prediction_line_imv
from .llm import CachedLLM
from .prompts_v3 import NO_EXPLANATION, parse_probability, render_chart

logger = logging.getLogger(__name__)
ARMS = ("empty", "static_llm", "taxai", "model_b", "trajectory_direction")
JUDGES = ("qwen2.5:14b", "phi4")

TURN1 = """You are an experienced ICU physician reviewing a patient with a clinical decision-support system.

PATIENT CHART:
{chart}

{prediction}

Based on the chart, estimate the probability (0-100%) that this patient will need invasive mechanical ventilation in the next 24 hours.
Answer with the percentage alone on the first line (for example: 35%), then give one brief reason."""

TURN2 = """The decision-support system also provides this explanation of its prediction:

{explanation}

Taking the explanation into account, re-estimate the probability (0-100%) that this patient will need invasive mechanical ventilation in the next 24 hours.
Answer with the percentage alone on the first line (for example: 35%), then give one brief reason."""


def components(s: np.ndarray, r: np.ndarray) -> Dict[str, float]:
    m = float(np.mean(s ** 2))
    i = float(2 * np.mean((s - s.mean()) * (r - r.mean())))
    d = float(2 * s.mean() * r.mean())
    return {"M": m, "I": i, "D": d, "dBrier": m - i - d}


def boot_mean(x: np.ndarray, seed: int = 42):
    rng = np.random.default_rng(seed)
    b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(1000)]
    return {"est": round(float(x.mean()), 5), "ci95": np.percentile(b, [2.5, 97.5]).round(5).tolist()}


def main() -> None:
    gdir = DATA_DIR / "v3" / "gate2_imv"
    meta = json.load(open(gdir / "gate2_imv.json"))
    thr = meta["threshold"]
    cases = pd.read_csv(gdir / "gate2_imv_cases.csv").drop_duplicates("hadm_id").set_index("hadm_id")
    gen = {}
    for line in open(gdir / "generation_cache.jsonl"):
        rec = json.loads(line)
        gen[(rec["method"], rec["hadm_id"])] = rec["response"]
    with open(DATA_DIR / "trajectories_test_v3.pkl", "rb") as f:
        trajs = pickle.load(f)
    out_dir = DATA_DIR / "v3" / "gate3_imv"
    out_dir.mkdir(parents=True, exist_ok=True)

    def arm_text(arm: str, hid: int, r) -> str:
        if arm == "empty":
            return NO_EXPLANATION
        if arm in ("static_llm", "taxai"):
            return gen[(arm, hid)][:3000]
        if arm == "model_b":
            return (f"A second model that additionally accounts for the timing and trajectory of the patient's "
                    f"recent changes estimates this risk at {r['pB']:.1%}.")
        direction = "higher" if r["pB"] > r["pA"] else "lower"
        return (f"Accounting for the timing and trajectory of recent changes, this patient's risk appears "
                f"{direction} than the AI estimate.")

    rows = []
    for judge in JUDGES:
        tag = judge.replace(":", "_").replace(".", "")
        llm = CachedLLM(out_dir / f"judge_cache_{tag}.jsonl")
        for k, (hid, r) in enumerate(cases.iterrows()):
            hid = int(hid)
            t1 = TURN1.format(chart=render_chart(trajs[hid]), prediction=prediction_line_imv(r["pA"], r["level"], thr))
            msgs = [{"role": "user", "content": t1}]
            r1 = llm.ask_chat(judge, "pre", "all", hid, msgs)
            pre = parse_probability(r1)
            for arm in ARMS:
                r2 = llm.ask_chat(judge, "post", arm, hid, msgs + [
                    {"role": "assistant", "content": r1},
                    {"role": "user", "content": TURN2.format(explanation=arm_text(arm, hid, r))}])
                rows.append({"judge": judge, "hadm_id": hid, "arm": arm, "level": r["level"], "y": int(r["y"]),
                             "pA": r["pA"], "pB": r["pB"], "pre": pre, "post": parse_probability(r2)})
            if (k + 1) % 25 == 0:
                logger.info(f"[{judge}] {k + 1}/{len(cases)}")
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "gate3_imv_cases.csv", index=False)

    res: Dict[str, object] = {}
    for judge, dj in df.groupby("judge"):
        dj = dj.dropna(subset=["pre", "post"])
        pre1 = dj.drop_duplicates("hadm_id")
        jr: Dict[str, object] = {"parse_ok": int(len(dj)),
                                 "anchoring_r": round(float(np.corrcoef(pre1["pre"], pre1["pA"])[0, 1]), 4),
                                 "pre_auroc": round(float(roc_auc_score(pre1["y"], pre1["pre"])), 4)}
        wide = {a: dj[dj["arm"] == a].set_index("hadm_id").sort_index() for a in ARMS}
        common = sorted(set.intersection(*[set(w.index) for w in wide.values()]))
        for a in ARMS:
            w = wide[a].loc[common]
            s, rr = (w["post"] - w["pre"]).values, (w["y"] - w["pre"]).values
            db = (w["post"] - w["y"]) ** 2 - (w["pre"] - w["y"]) ** 2
            jr[a] = {"dBrier": boot_mean(db.values), "components": {k: round(v, 5) for k, v in components(s, rr).items()},
                     "post_auroc": round(float(roc_auc_score(w["y"], w["post"])), 4),
                     "mean_shift": round(float(s.mean()), 4)}
            if a != "empty":
                e = wide["empty"].loc[common]
                diff = db.values - ((e["post"] - e["y"]) ** 2 - (e["pre"] - e["y"]) ** 2).values
                jr[a]["vs_empty"] = boot_mean(diff)
        res[judge] = jr
    p = res["qwen2.5:14b"]
    res["decision_primary_judge"] = {
        "gate3_open_informative_model_b": bool(p["model_b"]["vs_empty"]["ci95"][1] < 0),
        "gate3_open_direction_control": bool(p["trajectory_direction"]["vs_empty"]["ci95"][1] < 0),
        "static_llm_harmful": bool(p["static_llm"]["vs_empty"]["ci95"][0] > 0),
        "taxai_harmful": bool(p["taxai"]["vs_empty"]["ci95"][0] > 0),
    }
    json.dump(res, open(out_dir / "gate3_imv.json", "w"), indent=2)
    logger.info(json.dumps(res, indent=1))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()

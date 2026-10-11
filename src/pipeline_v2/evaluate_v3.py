"""v3 judge: own mortality estimate before and after the explanation.

Turn 1 (chart + AI prediction) is identical across arms and asked once per
case; turn 2 continues the same conversation with the arm's explanation.
Per-case rows store pre, post, outcome, and the parse status of each turn.

Usage:
    python -m src.pipeline_v2.evaluate_v3 --set pilot --judge qwen2.5:14b
"""

import argparse
import json
import logging
import pickle

import pandas as pd

from .config import DATA_DIR
from .llm import CachedLLM
from .prompts import prediction_line
from .prompts_v3 import (JUDGE_TURN1, JUDGE_TURN2, NO_EXPLANATION, parse_probability,
                         render_chart)

logger = logging.getLogger(__name__)
METHODS = ("empty", "random", "shap", "static_llm", "taxai")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["pilot", "main"], required=True)
    ap.add_argument("--judge", required=True)
    args = ap.parse_args()
    od = DATA_DIR / "v3" / args.set
    meta = json.load(open(od / "sample_meta.json"))
    thr, cases = meta["threshold"], meta["cases"]
    expl = json.load(open(od / "explanations.json"))
    with open(DATA_DIR / "trajectories_test_v3.pkl", "rb") as f:
        trajs = pickle.load(f)
    tag = args.judge.replace(":", "_").replace(".", "")
    llm = CachedLLM(od / f"judge_cache_{tag}.jsonl")

    rows = []
    for i, c in enumerate(cases):
        h = c["hadm_id"]
        turn1 = JUDGE_TURN1.format(chart=render_chart(trajs[h]),
                                   prediction=prediction_line(c["pred_proba"], c["level"], thr))
        msgs = [{"role": "user", "content": turn1}]
        r1 = llm.ask_chat(args.judge, "pre", "all", h, msgs)
        pre = parse_probability(r1)
        for m in METHODS:
            text = expl[m][str(h)] or NO_EXPLANATION
            r2 = llm.ask_chat(args.judge, "post", m, h, msgs + [
                {"role": "assistant", "content": r1},
                {"role": "user", "content": JUDGE_TURN2.format(explanation=text[:3000])}])
            post = parse_probability(r2)
            rows.append({"hadm_id": h, "method": m, "level": c["level"], "pred_proba": c["pred_proba"],
                         "mortality": c["mortality"], "pre": pre, "post": post,
                         "pre_parsed": pre is not None, "post_parsed": post is not None,
                         "prompt_chars": len(turn1) + len(text)})
        if (i + 1) % 25 == 0:
            logger.info(f"[{args.judge}] {i + 1}/{len(cases)} cases")
    pd.DataFrame(rows).to_csv(od / f"per_case_{tag}.csv", index=False)
    logger.info(f"Saved {len(rows)} rows ({args.set}, {args.judge})")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()

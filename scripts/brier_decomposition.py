"""Exact decomposition of the change in Brier score after an explanation (post hoc).

With shift s = post - pre and pre-explanation residual r = y - pre,
    dBrier = E[s^2] - 2 Cov(s, r) - 2 E[s] E[r]
           = movement cost (M) - information gain (I) - directional alignment (D).
An explanation improves accuracy only if I + D > M.

Usage:
    python scripts/brier_decomposition.py
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
RES = Path("results_v2/v3")
JUDGES = ("qwen25_14b", "phi4")
METHODS = ("empty", "shap", "static_llm", "taxai", "random")


def components(s: np.ndarray, r: np.ndarray) -> dict:
    m = float(np.mean(s ** 2))
    i = float(2 * np.mean((s - s.mean()) * (r - r.mean())))
    d = float(2 * s.mean() * r.mean())
    return {"M": m, "I": i, "D": d, "dBrier": m - i - d}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    rng = np.random.default_rng(42)
    out = {}
    for j in JUDGES:
        df = pd.read_csv(RES / f"per_case_{j}.csv")
        out[j] = {}
        for m in METHODS:
            d = df[df["method"] == m]
            s = (d["post"] - d["pre"]).values
            r = (d["mortality"] - d["pre"]).values
            est = components(s, r)
            check = float(np.mean((d["post"] - d["mortality"]) ** 2 - (d["pre"] - d["mortality"]) ** 2))
            assert abs(est["dBrier"] - check) < 1e-9
            boots = [components(s[i], r[i]) for i in (rng.integers(0, len(s), len(s)) for _ in range(2000))]
            out[j][m] = {k: {"est": round(v, 5),
                             "ci95": np.percentile([b[k] for b in boots], [2.5, 97.5]).round(5).tolist()}
                         for k, v in est.items()}
            logger.info(f"{j:11s} {m:10s} " + " ".join(f"{k}={v:+.4f}" for k, v in est.items()))
    json.dump(out, open(RES / "brier_decomposition.json", "w"), indent=2)


if __name__ == "__main__":
    main()

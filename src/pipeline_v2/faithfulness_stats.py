"""H4 (pre-specified): faithfulness of TAXAI vs Static LLM.

Wilcoxon signed-rank test on per-explanation supported-value rates, over
cases where both methods made at least one attributable numeric claim.
Random is reported as a negative control (another patient's values).

Usage:
    python -m src.pipeline_v2.faithfulness_stats
"""

import json
import logging

import pandas as pd
from scipy.stats import wilcoxon

from .config import DATA_DIR

logger = logging.getLogger(__name__)


def main() -> None:
    df = pd.read_csv(DATA_DIR / "cuef_v2" / "faithfulness_per_case.csv")
    summary = {}
    for m, d in df.groupby("method"):
        claims, sup = int(d["n_claims"].sum()), int(d["n_supported"].sum())
        with_claims = d[d["n_claims"] > 0]
        summary[m] = {
            "explanations": len(d), "with_claims": len(with_claims),
            "claims": claims, "supported": sup, "pooled_rate": round(sup / max(claims, 1), 4),
            "mean_rate": round(float(with_claims["rate"].mean()), 4),
            "frac_with_any_unsupported": round(float((with_claims["n_supported"] < with_claims["n_claims"]).mean()), 4),
        }
    w = df.pivot(index="hadm_id", columns="method", values="rate").dropna(subset=["taxai", "static_llm"])
    diff = w["taxai"] - w["static_llm"]
    p = float(wilcoxon(w["taxai"], w["static_llm"]).pvalue) if (diff != 0).any() else 1.0
    summary["H4_taxai_vs_static"] = {"n_pairs": len(w), "mean_diff": round(float(diff.mean()), 4),
                                     "median_diff": round(float(diff.median()), 4), "p_wilcoxon": p}
    with open(DATA_DIR / "cuef_v2" / "faithfulness_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    for k, v in summary.items():
        logger.info(f"{k}: {v}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()

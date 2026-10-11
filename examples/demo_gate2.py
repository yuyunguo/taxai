"""Run the registered Gate 2 scoring and H1 rule on the synthetic example.

Uses the same verdict parser, bootstrap, and decision rule as the registered
analysis (src/pipeline_v2/gate2_v4_analysis.py).

Usage (repository root): python examples/make_synthetic.py && python examples/demo_gate2.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline_v2.gate2_v4_analysis import SCORE, boot_auroc, gate2_status, load_cache, verdict  # noqa: E402


def main() -> None:
    cases = pd.read_csv("examples/synthetic_imv_sample.csv")
    cache = load_cache(Path("examples/synthetic_gen_imv.jsonl"))
    y = cases["y"].values
    for cond in ("static_llm", "taxai"):
        v = pd.Series([verdict(cache[(cond, int(h))]) for h in cases["hadm_id"]])
        a = boot_auroc(y, v.map(SCORE).fillna(0).values)
        print(f"{cond:10s} AUROC {a['est']:.3f} 95% CI {a['ci95']} -> {gate2_status(a)}")
    rc = boot_auroc(y, cases["rule_score"].values)
    print(f"rule control AUROC {rc['est']:.3f} 95% CI {rc['ci95']}")


if __name__ == "__main__":
    main()

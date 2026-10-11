"""Generate a synthetic Gate 2 input in the real file formats (no patient data).

Writes examples/synthetic_imv_sample.csv (same columns as data_v2/v4/imv_sample.csv)
examples/synthetic_gen_imv.jsonl (same records as the explanation caches), and
examples/synthetic_gen_tierC.jsonl (elicited-probability responses, addendum C1a).
All identifiers, probabilities, outcomes, and texts are random and synthetic.

Usage (repository root): python examples/make_synthetic.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("examples")
VERDICTS = ("too high", "about right", "too low")


def main(seed: int = 42, n_per_class: int = 200) -> None:
    rng = np.random.default_rng(seed)
    n = 2 * n_per_class
    y = np.repeat([1, 0], n_per_class)
    pa = np.clip(rng.beta(2, 60, n) + 0.01 * y, 0.001, 0.0646)
    pb = np.clip(pa * np.exp(rng.normal(0.25 * y, 0.5, n)), 0.001, 0.99)
    df = pd.DataFrame({"hadm_id": 900000001 + np.arange(n), "y": y,
                       "split": rng.choice(["val", "test"], n), "pA": pa.round(4), "pB": pb.round(4),
                       "level": "LOW", "in_prior_sample": 0, "rule_score": np.where(pb > pa, 1, -1)})
    df.to_csv(OUT / "synthetic_imv_sample.csv", index=False)
    with open(OUT / "synthetic_gen_imv.jsonl", "w") as fh:
        for h, yy in zip(df["hadm_id"], y):
            for method in ("static_llm", "taxai"):
                p = [0.25, 0.5, 0.25] if not yy else [0.2, 0.5, 0.3]
                v = VERDICTS[rng.choice(3, p=p)]
                text = f"Synthetic explanation. Overall, the AI estimate appears {v} for this patient."
                fh.write(json.dumps({"method": method, "hadm_id": int(h), "response": text}) + "\n")
    with open(OUT / "synthetic_gen_tierC.jsonl", "w") as fh:
        for h, p_a, yy in zip(df["hadm_id"], df["pA"], y):
            p = float(np.clip(p_a * np.exp(rng.normal(0.2 * yy, 0.6)), 0.001, 0.99)) * 100
            text = ("Synthetic explanation. My estimate of the probability that this patient will need invasive "
                    f"mechanical ventilation in the next 24 hours is {p:.1f}%.")
            fh.write(json.dumps({"method": "C1a", "hadm_id": int(h), "response": text}) + "\n")


if __name__ == "__main__":
    main()

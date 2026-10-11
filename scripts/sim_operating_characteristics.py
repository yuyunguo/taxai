"""Operating characteristics of the explainer-ceiling (Gate 2) decision rule (post hoc).

Simulates three-level verdicts (too high < about right < too low) for samples of
200 events and 200 non-events, with a fixed overall verdict mix (the explainer's
disposition) and a true verdict AUROC set by a latent shift. Each simulated
sample is classified with the registered H1 rule: case-level percentile
bootstrap (1,000 resamples); open if AUROC >= 0.60 with 95% and Bonferroni 99%
lower limits above 0.5; closed if the 95% upper limit is below 0.60;
inconclusive otherwise. No patient data are used.

Usage:
    python scripts/sim_operating_characteristics.py
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SimConfig:
    n_events: int = 200
    n_nonevents: int = 200
    n_sims: int = 2000
    n_boot: int = 1000
    threshold: float = 0.60
    seed: int = 42
    targets: Tuple[float, ...] = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75)
    out: Path = Path("results_v2/v4/sim_operating_characteristics.json")


# Overall verdict shares (too high, about right, too low). The three observed mixes
# are the TAXAI ventilation LOW distributions; "balanced" is a hypothetical reference.
PATTERNS: Dict[str, Tuple[float, float, float]] = {
    "balanced (hypothetical)": (0.25, 0.50, 0.25),
    "one-directional (as qwen2.5:14b)": (301 / 400, 2 / 400, 97 / 400),
    "frequent dissent (as gemma3:27b)": (122 / 400, 276 / 400, 2 / 400),
    "deferential (as qwen2.5:32b)": (0.001, 380 / 400, 20 / 400 - 0.001),
}


def class_shares(delta: float, marg: Tuple[float, float, float]) -> Tuple[np.ndarray, np.ndarray]:
    """Event and non-event verdict shares when latent L = delta*y + N(0,1) is cut at pooled quantiles."""
    def mix_cdf(x: float) -> float:
        return 0.5 * norm.cdf(x) + 0.5 * norm.cdf(x - delta)

    c1 = brentq(lambda x: mix_cdf(x) - marg[0], -40, 40)
    c2 = brentq(lambda x: mix_cdf(x) - (marg[0] + marg[1]), -40, 40)
    non = np.array([norm.cdf(c1), norm.cdf(c2) - norm.cdf(c1), 1 - norm.cdf(c2)])
    ev = np.array([norm.cdf(c1 - delta), norm.cdf(c2 - delta) - norm.cdf(c1 - delta), 1 - norm.cdf(c2 - delta)])
    return ev, non


def auroc_counts(e: np.ndarray, n: np.ndarray) -> np.ndarray:
    """AUROC of ordinal verdicts from counts; columns are (too high, about right, too low)."""
    ne, nn = e.sum(-1), n.sum(-1)
    greater = e[..., 1] * n[..., 0] + e[..., 2] * (n[..., 0] + n[..., 1])
    ties = (e * n).sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (greater + 0.5 * ties) / (ne * nn)


def true_auroc(delta: float, marg: Tuple[float, float, float]) -> float:
    ev, non = class_shares(delta, marg)
    return float(auroc_counts(ev, non))


def solve_delta(target: float, marg: Tuple[float, float, float]) -> Optional[float]:
    if abs(target - 0.5) < 1e-9:
        return 0.0
    hi = true_auroc(12.0, marg)
    if target >= hi:
        return None
    return brentq(lambda d: true_auroc(d, marg) - target, 0.0, 12.0)


def classify(cfg: SimConfig, ev: np.ndarray, non: np.ndarray, rng: np.random.Generator) -> Dict[str, float]:
    counts = {"open": 0, "closed": 0, "inconclusive": 0}
    for _ in range(cfg.n_sims):
        e = rng.multinomial(cfg.n_events, ev)
        n = rng.multinomial(cfg.n_nonevents, non)
        est = auroc_counts(e, n)
        cells = np.concatenate([e, n]).astype(float)
        boot = rng.multinomial(cfg.n_events + cfg.n_nonevents, cells / cells.sum(), size=cfg.n_boot)
        b = auroc_counts(boot[:, :3], boot[:, 3:])
        b = b[np.isfinite(b)]
        lo95, hi95 = np.percentile(b, [2.5, 97.5])
        lo99 = np.percentile(b, 0.5)
        if est >= cfg.threshold and lo95 > 0.5 and lo99 > 0.5:
            counts["open"] += 1
        elif hi95 < cfg.threshold:
            counts["closed"] += 1
        else:
            counts["inconclusive"] += 1
    return {k: round(v / cfg.n_sims, 3) for k, v in counts.items()}


def main(cfg: SimConfig = SimConfig()) -> None:
    rng = np.random.default_rng(cfg.seed)
    res: Dict[str, object] = {"config": {k: (str(v) if isinstance(v, Path) else v)
                                         for k, v in cfg.__dict__.items()}, "patterns": {}}
    for name, marg in PATTERNS.items():
        rows: Dict[str, object] = {"verdict_mix": [round(x, 4) for x in marg],
                                   "max_attainable_auroc": round(true_auroc(12.0, marg), 3)}
        for t in cfg.targets:
            d = solve_delta(t, marg)
            if d is None:
                rows[f"{t:.2f}"] = "unattainable"
                continue
            ev, non = class_shares(d, marg)
            rows[f"{t:.2f}"] = classify(cfg, ev, non, rng)
            logger.info(f"{name} AUROC {t:.2f}: {rows[f'{t:.2f}']}")
        res["patterns"][name] = rows
    cfg.out.write_text(json.dumps(res, indent=1))
    logger.info(f"wrote {cfg.out}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()

"""Sensitivity analysis: Tier C endpoints with 10,000 bootstrap resamples.

Runs the hashed analysis (tierC_v4_analysis.py) unchanged except for N_BOOT, and redirects
its output to tierC_v4_boot10k.json so the primary tierC_v4.json is never opened for writing.
Not part of the pre-specified addendum; reported as a robustness check.

Usage (project root): python -m src.pipeline_v2.tierC_v4_boot10k
"""
import builtins
import logging
from pathlib import Path
from typing import Any

from . import tierC_v4_analysis as analysis

logger = logging.getLogger(__name__)

N_BOOT_SENS = 10_000
PRIMARY_NAME = "tierC_v4.json"
SENS_NAME = "tierC_v4_boot10k.json"


def _redirecting_open(file: Any, mode: str = "r", *args: Any, **kwargs: Any):
    """Send writes aimed at the primary result file to the sensitivity file instead."""
    path = Path(file)
    if path.name == PRIMARY_NAME and "w" in mode:
        path = path.with_name(SENS_NAME)
    return builtins.open(path, mode, *args, **kwargs)


def main() -> None:
    analysis.N_BOOT = N_BOOT_SENS
    analysis.open = _redirecting_open  # module globals shadow builtins.open
    logger.info("Tier C sensitivity: N_BOOT=%d -> %s", N_BOOT_SENS, SENS_NAME)
    analysis.main()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()

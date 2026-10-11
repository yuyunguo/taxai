"""Tier C prompt variants (plan/prereg-v4-addendum-tierC.md, section 2).

Every variant is derived from the registered ventilation instruction (EXPLAIN_IMV) by exact
substitution; each substituted phrase is asserted to exist, so a variant cannot drift silently.
"""

import re
from typing import Optional

from .gate2_imv import EXPLAIN_IMV

_VERDICT_LINE = ('4. End with exactly one sentence: "Overall, the AI estimate appears '
                 '[too high / about right / too low] for this patient."')
_ELICIT_LINE = ('4. End with exactly one sentence: "My estimate of the probability that this patient will '
                'need invasive mechanical ventilation in the next 24 hours is X%."')
_JOB = "Your job is to help the clinician judge whether this estimate fits the patient. Do not defend the prediction."
_LEVEL = "2. State which findings support a {level} risk estimate and which argue against it."
BASE_RATE = ("Among comparable ICU patients, about 2% start invasive mechanical ventilation within the "
             "next 24 hours.")


def _sub(text: str, old: str, new: str) -> str:
    assert old in text, f"phrase not found in registered instruction: {old[:60]}"
    return text.replace(old, new)


# C1a: elicited probability, prediction shown.
ELICIT_WITH_PRED = _sub(EXPLAIN_IMV, _VERDICT_LINE, _ELICIT_LINE)

# C1b: elicited probability, prediction hidden (no prediction line, no label).
ELICIT_NO_PRED = _sub(_sub(_sub(EXPLAIN_IMV, _VERDICT_LINE, _ELICIT_LINE),
                           "{prediction}\n" + _JOB, "Your job is to help the clinician judge this patient's risk."),
                      _LEVEL, "2. State which findings point toward higher risk and which point toward lower risk.")

# C3: registered verdict, "Do not defend the prediction" removed, base rate stated after the prediction line.
ALT_INSTRUCTION = _sub(EXPLAIN_IMV, "{prediction}\n" + _JOB,
                       "{prediction}\n" + BASE_RATE + "\nYour job is to help the clinician judge whether this "
                       "estimate fits the patient.")

_PCT = re.compile(r"(\d+(?:\.\d+)?)\s*%")


def parse_elicited(resp: Optional[str]) -> Optional[float]:
    """Last percentage in the final sentence, else in the whole response; None if absent or out of range."""
    if not resp:
        return None
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", resp.strip()) if s]
    for text in ((sentences[-1] if sentences else ""), resp):
        hits = _PCT.findall(text)
        if hits:
            v = float(hits[-1])
            return v / 100 if 0 <= v <= 100 else None
    return None

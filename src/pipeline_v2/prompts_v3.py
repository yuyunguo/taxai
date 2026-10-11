"""v3 prompt templates.

* The chart (full chronological record) is rendered identically for every
  arm and shown to the judge.
* The generation instruction is shared by Static LLM and TAXAI and ends with
  a directional verdict (too high / about right / too low) instead of v2's
  consistency verdict.
* The judge gives its own mortality estimate before (turn 1) and after
  (turn 2) seeing the explanation.
"""

import re
from typing import Dict, Optional

from .prompts import TIMELINE_LABELS, render_context, render_summary  # noqa: F401 (re-export)


def render_chart(traj: Dict) -> str:
    lines = [render_context(traj), "", "CHRONOLOGICAL RECORD (hours relative to ICU admission):"]
    for e in traj["events"]:
        line = f"ICU hour {e['t']:+.0f}: {TIMELINE_LABELS[e['type']]}: {e['label']}"
        if e["value"] is not None:
            line += f" = {e['value']:g}" + (f" {e['uom']}" if e["uom"] else "")
        lines.append(line)
    if len(lines) == 3:
        lines.append("(no events recorded)")
    return "\n".join(lines)


EXPLAIN_V3 = """You are a clinical decision-support assistant in an intensive care unit.
{prediction}
Your job is to help the clinician judge whether this estimate fits the patient. Do not defend the prediction.

{data}

Write 3-5 sentences that:
1. Cite the specific findings, with values and times, that are most relevant to mortality risk.
2. State which findings support a {level} risk estimate and which argue against it.
3. Mention important missing or sparse data if relevant.
4. End with exactly one sentence: "Overall, the AI estimate appears [too high / about right / too low] for this patient."
Use only the data shown. Do not state diagnoses that are not supported by the data.

Explanation:"""

JUDGE_TURN1 = """You are an experienced ICU physician reviewing a patient with a clinical decision-support system.

PATIENT CHART:
{chart}

{prediction}

Based on the chart, estimate the probability (0-100%) that this patient dies during this hospitalization.
Answer with the percentage alone on the first line (for example: 35%), then give one brief reason."""

JUDGE_TURN2 = """The decision-support system also provides this explanation of its prediction:

{explanation}

Taking the explanation into account, re-estimate the probability (0-100%) that this patient dies during this hospitalization.
Answer with the percentage alone on the first line (for example: 35%), then give one brief reason."""

NO_EXPLANATION = "(No explanation provided.)"

_PCT = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_NUM = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\w.])")


def parse_probability(resp: str) -> Optional[float]:
    """Probability in [0, 1] from the first line (percent preferred), else from the whole text."""
    for text in (resp.strip().splitlines()[0] if resp.strip() else "", resp):
        m = _PCT.search(text)
        if m:
            return min(max(float(m.group(1)) / 100, 0.0), 1.0)
        m = _NUM.search(text)
        if m:
            v = float(m.group(1))
            return min(max(v if v <= 1 else v / 100, 0.0), 1.0)
    return None

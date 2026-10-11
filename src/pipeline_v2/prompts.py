"""Text rendering of trajectories and all prompt templates (v2).

Explanation prompts ask for an *assessment* of the prediction (evidence for
and against), not a justification. TAXAI and Static LLM share the identical
instruction text; they differ only in whether the chronological timeline is
provided. Judge prompts never show ICD codes or the patient's outcome.
"""

from typing import Dict, List

TIMELINE_LABELS = {"lab": "Lab", "vital": "Vital", "med": "Medication started", "note": "Imaging"}


def render_context(traj: Dict) -> str:
    c = traj["context"]
    sex = {"M": "male", "F": "female"}.get(c["sex"], c["sex"])
    return (f"Patient: {c['age']}-year-old {sex}; admission type: {c['admission_type']}; "
            f"ICU: {c['icu_unit']}. Data cover up to 24 h before and the first 24 h after ICU admission.")


def render_timeline(traj: Dict) -> str:
    """Chronological event list; hours are relative to ICU admission."""
    lines: List[str] = []
    for e in traj["events"]:
        head = f"ICU hour {e['t']:+.0f}: {TIMELINE_LABELS[e['type']]}: {e['label']}"
        if e["value"] is not None:
            head += f" = {e['value']:g}" + (f" {e['uom']}" if e["uom"] else "")
        lines.append(head)
    return "\n".join(lines) if lines else "(no events recorded)"


def render_summary(traj: Dict) -> str:
    """Aggregate summary: first/last lab values, vital ranges, medication list, imaging count."""
    ev = traj["events"]
    parts: List[str] = [render_context(traj)]
    labs: Dict[str, list] = {}
    vitals: Dict[str, list] = {}
    for e in ev:
        if e["type"] == "lab":
            labs.setdefault(e["label"], []).append((e["t"], e["value"]))
        elif e["type"] == "vital":
            vitals.setdefault(e["label"], []).append((e["t"], e["value"]))
    for name, vals in labs.items():
        vals.sort()
        if len(vals) >= 2:
            (t0, v0), (t1, v1) = vals[0], vals[-1]
            word = "rose" if v1 > v0 else "fell" if v1 < v0 else "unchanged"
            parts.append(f"{name}: {v0:g} -> {v1:g} ({word}, over {t1 - t0:.0f} h)")
        else:
            parts.append(f"{name}: {vals[0][1]:g}")
    for name, vals in vitals.items():
        v = [x for _, x in vals]
        parts.append(f"{name}: range {min(v):g}-{max(v):g}")
    meds = [e["label"] for e in ev if e["type"] == "med"]
    if meds:
        parts.append("Medications started: " + ", ".join(meds[:8]))
    n_img = sum(e["type"] == "note" for e in ev)
    if n_img:
        parts.append(f"Imaging reports: {n_img}")
    return "\n".join(parts)


def prediction_line(p: float, level: str, thr: float) -> str:
    return (f"AI MORTALITY PREDICTION: {p:.1%} risk of in-hospital death "
            f"(labeled {level} risk; the system's HIGH-risk threshold is {thr:.0%}).")


EXPLAIN = """You are a clinical decision-support assistant in an intensive care unit.
{prediction}
Your job is to help the clinician judge whether this estimate fits the patient. Do not defend the prediction.

{data}

Write 3-5 sentences that:
1. Cite the specific findings, with values and times, that are most relevant to mortality risk.
2. State which findings support a {level} risk estimate and which argue against it.
3. Mention important missing or sparse data if relevant.
4. End with exactly one sentence: "Overall, the data are [consistent / partly consistent / inconsistent] with a {level} risk estimate."
Use only the data shown. Do not state diagnoses that are not supported by the data.

Explanation:"""

PATIENT = """Explain an AI risk assessment to a patient's family in plain language (no jargon, no numbers).
{prediction}

{data}

Write 2-3 sentences: what the care team is watching, whether the patient seems to be improving or getting worse, and that the assessment is uncertain.

Explanation:"""

REGULATOR = """Produce a structured audit record for an AI mortality prediction.
{prediction}

{data}

Use exactly these headings:
- Prediction and threshold
- Key inputs (each with its value and ICU hour, copied from the data)
- Evidence for and against the risk label
- Data completeness (which measurements are missing or sparse)
Do not add diagnoses or codes that are not in the data.

Audit record:"""

JUDGE_HEADER = """You are an experienced ICU physician reviewing a patient with a clinical decision-support system.

PATIENT DATA:
{summary}

{prediction}

AI EXPLANATION:
{explanation}
"""

DC_TASK = JUDGE_HEADER + """
List the 3 most likely diagnoses for this hospitalization, ranked by likelihood.
Respond in EXACTLY this format (one per line):
1. [diagnosis]
2. [diagnosis]
3. [diagnosis]"""

TAS_TASK = JUDGE_HEADER + """
List the top 5 treatment or monitoring actions you would order for this patient now.
Respond in EXACTLY this format (one per line):
1. [action]
2. [action]
3. [action]
4. [action]
5. [action]"""

RELIANCE_TASK = JUDGE_HEADER + """
Do you agree with the AI's {level}-risk classification for this patient?
A) ACCEPT the AI classification
B) OVERRIDE the AI classification
Answer with "A" or "B" on the first line, then give one brief reason."""

NO_EXPLANATION = "(No explanation provided.)"

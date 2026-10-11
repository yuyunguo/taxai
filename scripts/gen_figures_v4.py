"""Figures for the two-ceiling audit (manuscript v4).

Figure 1 (schematic): the information ceiling (Gate 1) and the explainer
ceiling (Gate 2), with the positive controls that validate the audit.
Figure 2 (main): a, Gate 1 dAUROC (model B - model A) for three outcomes;
b, Gate 2 verdict AUROC in ventilation LOW per explainer, with the rule
control and model benchmarks. Panel b reads results_v2/v4/gate2_v4.json and
is drawn as "pending" until the registered runs finish.
Figure 3 (failure modes, H4): verdict composition by outcome per explainer,
from results_v2/v4/h4_verdict_counts.json (aggregate counts only).

Usage:
    python scripts/gen_figures_v4.py
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

logger = logging.getLogger(__name__)
RES = Path("results_v2")
OUT = Path("figures_v2")
RED, GREY, BLUE, GREEN = "#C0392B", "#9E9E9E", "#4C8CBF", "#5BAE6E"
EXPLAINERS: Tuple[Tuple[str, str], ...] = (
    ("qwen2.5:14b", "Qwen2.5 14B"), ("qwen2.5:32b", "Qwen2.5 32B"),
    ("qwen2.5:72b-instruct-q3_K_M", "Qwen2.5 72B"), ("gemma3:27b", "Gemma 3 27B"),
    ("gpt-oss:20b", "gpt-oss 20B"),
)


def figure1() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.0), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.15, 4.2)
    ax.axis("off")

    def box(x: float, y: float, w: float, h: float, text: str, fc: str, ec: str = "black",
            fs: float = 6.3, weight: str = "normal") -> None:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.08", fc=fc, ec=ec, lw=0.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight=weight)

    def arrow(x1: float, y1: float, x2: float, y2: float) -> None:
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=8, lw=0.8, color="black"))

    box(0.05, 2.0, 1.5, 1.3, "Patient\ntrajectory\n+ model\nprediction", "#F2F2F2", fs=6.0, weight="bold")
    box(2.0, 2.75, 3.0, 0.65, "Information ceiling (Gate 1)", "#FBE9E7", ec=RED, fs=7, weight="bold")
    box(2.0, 1.85, 3.0, 0.8, "Does the trajectory hold outcome\ninformation the prediction has not used?", "white", fs=5.6)
    box(2.0, 1.05, 3.0, 0.7, "Test (no LLM): model A vs. model B\n(+ trajectory dynamics), ΔAUROC", "#F7F7F7", ec=GREY, fs=5.6)
    box(5.5, 2.75, 3.0, 0.65, "Explainer ceiling (Gate 2)", "#FBE9E7", ec=RED, fs=7, weight="bold")
    box(5.5, 1.85, 3.0, 0.8, "Does the explainer's verdict separate\nright from wrong predictions?", "white", fs=5.6)
    box(5.5, 1.05, 3.0, 0.7, "Test (no reader): verdict AUROC\nwithin label strata", "#F7F7F7", ec=GREY, fs=5.6)
    box(5.5, 0.0, 3.0, 0.95, "Positive control: rule explainer,\nsign(pB − pA)\nGiven-signal test: LLM given\nmodel B's estimate (hint)",
        "#E8F0F8", ec=BLUE, fs=5.3)
    box(9.0, 2.0, 0.95, 1.3, "Worth a\nreader\nstudy", "#E8F4EA", fs=6.0, weight="bold")
    arrow(1.55, 2.65, 2.0, 2.65)
    arrow(5.0, 2.65, 5.5, 2.65)
    arrow(8.5, 2.65, 9.0, 2.65)
    ax.text(5.25, 3.75, "Both ceilings must be open before an explanation can improve a reader's judgment",
            ha="center", fontsize=6.8, style="italic")
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "fig1_two_ceilings.png", bbox_inches="tight", dpi=600)
    plt.close(fig)


def _gate1_rows() -> List[Tuple[str, float, float, float]]:
    mort = json.load(open(RES / "info_ceiling.json"))["auroc"]
    other = json.load(open(RES / "gate1_other_outcomes.json"))
    rows = [("In-hospital death", mort["diff"], *mort["ci95"])]
    for key, label in (("invasive_ventilation", "Invasive ventilation"), ("norepinephrine", "Norepinephrine start")):
        a = other[key]["auroc"]
        rows.append((label, a["diff"], *a["ci95"]))
    return rows


def _gate2_rows(res: Dict[str, object]) -> List[Tuple[str, float, float, float, str]]:
    rows = []
    for key, label in EXPLAINERS:
        r = res.get(key)
        if isinstance(r, dict):
            a = r["LOW"]["taxai"]["auroc"]
            rows.append((label, a["est"], a["ci95"][0], a["ci95"][1], r["gate2_status_taxai"]))
    rc = res.get("rule_control_LOW")
    if rc:
        rows.append(("Rule control", rc["est"], rc["ci95"][0], rc["ci95"][1], "control"))
    return rows


def figure2() -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.6), dpi=300, gridspec_kw={"width_ratios": [1, 1.25]})
    rows = _gate1_rows()
    for i, (label, d, lo, hi) in enumerate(rows):
        color = GREEN if lo > 0 and d >= 0.010 else GREY
        a.errorbar(d, i, xerr=[[d - lo], [hi - d]], fmt="o", color=color, ms=4, capsize=2, lw=1)
    a.axvline(0, color="black", lw=0.6)
    a.axvline(0.010, color=RED, lw=0.6, ls="--")
    a.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=6.5)
    a.invert_yaxis()
    a.set_xlabel("ΔAUROC, model B − model A (95% CI)", fontsize=6.5)
    a.set_title("a  Information ceiling (Gate 1)", loc="left", fontsize=7.5, weight="bold")
    a.tick_params(labelsize=6)

    path = RES / "v4" / "gate2_v4.json"
    b.set_title("b  Explainer ceiling (Gate 2), ventilation LOW", loc="left", fontsize=7.5, weight="bold")
    b.axvspan(0.3, 0.60, color="#F3F3F3", zorder=0)
    b.axvline(0.5, color="black", lw=0.6)
    b.axvline(0.60, color=RED, lw=0.6, ls="--")
    b.set_xlim(0.3, 1.0)
    b.set_xlabel("Verdict AUROC for the event (95% CI)", fontsize=6.5)
    b.tick_params(labelsize=6)
    if not path.exists():
        b.text(0.65, 0.5, "Pending registered runs", transform=b.transAxes, ha="center", fontsize=7, color=GREY)
        b.set_yticks([])
    else:
        res = json.load(open(path))["imv"]
        g2 = _gate2_rows(res)
        for i, (label, est, lo, hi, status) in enumerate(g2):
            color = BLUE if status == "control" else (GREEN if status == "open" else RED if status == "closed" else GREY)
            b.errorbar(est, i, xerr=[[est - lo], [hi - est]], fmt="o", color=color, ms=4, capsize=2, lw=1)
        b.set_yticks(range(len(g2)), [r[0] for r in g2], fontsize=6.5)
        b.set_ylim(len(g2) - 0.5, -1.0)  # inverted, with headroom for reference labels
        ref = next(v["LOW"] for v in res.values() if isinstance(v, dict) and "LOW" in v)
        for key, label, ls, dx, ha in (("benchmark_pA", "model A", ":", -0.006, "right"),
                                       ("benchmark_pB", "model B", "-.", 0.006, "left")):
            x = ref[key]["est"]
            b.axvline(x, color=GREY, lw=0.8, ls=ls)
            b.text(x + dx, -0.65, f"{label} {x:.2f}", fontsize=5.5, color=GREY, ha=ha, va="center")
    fig.tight_layout()
    fig.savefig(OUT / "fig2_two_ceilings_evidence.png", bbox_inches="tight", dpi=600)
    plt.close(fig)


def figure3() -> None:
    """Failure modes (H4): verdict composition by outcome, ventilation LOW, per explainer."""
    path = RES / "v4" / "h4_verdict_counts.json"
    if not path.exists():
        logger.warning(f"{path} missing; Figure 3 skipped")
        return
    counts = json.load(open(path))
    rows = [("qwen2.5:14b", "taxai", "Qwen2.5 14B"), ("qwen2.5:14b", "taxai_hint", "Qwen2.5 14B + hint"),
            ("gemma3:27b", "taxai", "Gemma 3 27B"), ("qwen2.5:32b", "taxai", "Qwen2.5 32B"),
            ("gpt-oss:20b", "taxai", "gpt-oss 20B"), ("qwen2.5:72b-instruct-q3_K_M", "taxai", "Qwen2.5 72B")]
    cats = (("too low", GREEN, "too low (would rank ventilated patients higher)"), ("about right", "#D9D9D9", "about right"),
            ("too high", RED, "too high"), ("missing", "white", "missing"))
    fig, ax = plt.subplots(figsize=(7.2, 3.0), dpi=300)
    ylabels, y = [], 0.0
    for model, cond, label in rows:
        c = counts[f"{model}|LOW|{cond}"]
        for outcome, tag in (("1", "ventilated"), ("0", "not ventilated")):
            n = sum(c[outcome].values())
            left = 0.0
            for key, color, _ in cats:
                w = c[outcome].get(key, 0) / n
                ax.barh(y, w, left=left, color=color, edgecolor="black", lw=0.3, height=0.8)
                left += w
            k = c[outcome].get("too low", 0)
            ax.text(1.01, y, f"{k}/{n}", va="center", fontsize=5.5, color=GREEN if outcome == "1" else GREY)
            ylabels.append((y, f"{label} \u2014 {tag}" if outcome == "1" else tag))
            y += 0.9
        y += 0.5
    ax.set_yticks([t for t, _ in ylabels], [lab for _, lab in ylabels], fontsize=5.8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of verdicts (TAXAI condition; ventilation, LOW labels)", fontsize=6.5)
    ax.tick_params(axis="x", labelsize=6)
    ax.text(1.01, -0.9, '"too low"', fontsize=5.5, color=GREY)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=col, ec="black", lw=0.3) for _, col, _ in cats]
    ax.legend(handles, [lab for _, _, lab in cats], fontsize=5.8, ncol=4, frameon=False,
              loc="upper center", bbox_to_anchor=(0.5, 1.12))
    fig.savefig(OUT / "fig3_failure_modes.png", bbox_inches="tight", dpi=600)
    plt.close(fig)


def main() -> None:
    figure1()
    figure2()
    figure3()
    logger.info(f"Figures written to {OUT}/")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()

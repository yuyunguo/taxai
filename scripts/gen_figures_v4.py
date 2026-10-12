"""Figures for the two-ceiling audit (manuscript v4).

Figure 1 (schematic): the information ceiling (Gate 1) and the explainer
ceiling (Gate 2), with the positive controls that validate the audit.
Figure 2 (main): a, Gate 1 dAUROC (model B - model A) for three outcomes;
b, Gate 2 verdict AUROC in ventilation LOW per explainer, with the rule
control and model benchmarks. Panel b reads results_v2/v4/gate2_v4.json and
is drawn as "pending" until the registered runs finish.
Figure 2c: readout capacity, observed verdict AUROC against the highest AUROC
attainable from each explainer's verdict counts (results_v2/v4/posthoc_v4_gate2.json).
Figure 3 (failure modes, H4): verdict composition by outcome per explainer,
from results_v2/v4/h4_verdict_counts.json (aggregate counts only).
Figure 4 (pre-specified addendum): a, AUROC by readout per explainer (verdict,
probability with the prediction shown, probability with it hidden); b, anchoring,
AUROC against the Spearman correlation with model A (results_v2/v4/tierC_v4.json).

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
    ("qwen2.5:14b", "qwen2.5:14b"), ("qwen2.5:32b", "qwen2.5:32b"),
    ("qwen2.5:72b-instruct-q3_K_M", "qwen2.5:72b"), ("gemma3:27b", "gemma3:27b"),
    ("gpt-oss:20b", "gpt-oss:20b"),
)


def figure1() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.0), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.15, 4.2)
    ax.axis("off")

    def box(x: float, y: float, w: float, h: float, text: str, fc: str, ec: str = "black",
            fs: float = 7.2, weight: str = "normal") -> None:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.08", fc=fc, ec=ec, lw=0.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight=weight)

    def arrow(x1: float, y1: float, x2: float, y2: float) -> None:
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=8, lw=0.8, color="black"))

    box(0.05, 2.0, 1.5, 1.3, "Patient\ntrajectory\n+ model\nprediction", "#F2F2F2", fs=6.9, weight="bold")
    box(2.0, 2.75, 3.0, 0.65, "Information ceiling (Gate 1)", "#FBE9E7", ec=RED, fs=7.4, weight="bold")
    box(2.0, 1.85, 3.0, 0.8, "Does a trajectory representation\nadd recoverable information\nthe prediction has not used?", "white", fs=6.0)
    box(2.0, 1.05, 3.0, 0.7, "Test (no LLM): model A vs. model B\n(+ trajectory dynamics), ΔAUROC", "#F7F7F7", ec=GREY, fs=6.1)
    box(5.5, 2.75, 3.0, 0.65, "Explainer ceiling (Gate 2)", "#FBE9E7", ec=RED, fs=7.4, weight="bold")
    box(5.5, 1.85, 3.0, 0.8, "Does the explainer's verdict\nseparate right from\nwrong predictions?", "white", fs=6.0)
    box(5.5, 1.05, 3.0, 0.7, "Test (no reader): verdict AUROC\nwithin label strata", "#F7F7F7", ec=GREY, fs=6.1)
    box(5.5, 0.0, 3.0, 0.95, "Positive control: rule explainer,\nsign(pB − pA)\nGiven-signal test: LLM given\nmodel B's estimate (hint)",
        "#E8F0F8", ec=BLUE, fs=6.2)
    box(9.0, 2.0, 0.95, 1.3, "Proceed to\nreader\nevaluation", "#E8F4EA", fs=6.9, weight="bold")
    arrow(1.55, 2.65, 2.0, 2.65)
    arrow(5.0, 2.65, 5.5, 2.65)
    arrow(8.5, 2.65, 9.0, 2.65)
    ax.text(5.25, 3.75, "Both ceilings must be open for an explanation to improve discrimination of prediction errors",
            ha="center", fontsize=7.8, style="italic")
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
    fig = plt.figure(figsize=(7.2, 4.6), dpi=300)
    gs = fig.add_gridspec(2, 2, height_ratios=[0.8, 1.6], width_ratios=[1.45, 1], hspace=0.55, wspace=0.06)
    a = fig.add_subplot(gs[0, :])
    b = fig.add_subplot(gs[1, 0])
    c = fig.add_subplot(gs[1, 1], sharey=b)
    rows = _gate1_rows()
    for i, (label, d, lo, hi) in enumerate(rows):
        color = GREEN if lo > 0 and d >= 0.010 else GREY
        a.errorbar(d, i, xerr=[[d - lo], [hi - d]], fmt="o", color=color, ms=4, capsize=2, lw=1)
    a.axvline(0, color="black", lw=0.6)
    a.axvline(0.010, color=RED, lw=0.6, ls="--")
    a.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=7.5)
    a.invert_yaxis()
    a.set_xlabel("ΔAUROC, model B − model A (95% CI)", fontsize=7.5)
    a.set_title("a  Information ceiling (Gate 1)", loc="left", fontsize=8.5, weight="bold")
    a.tick_params(labelsize=6)

    path = RES / "v4" / "gate2_v4.json"
    b.set_title("b  Explainer ceiling (Gate 2)", loc="left", fontsize=8.5, weight="bold")
    b.axvspan(0.3, 0.60, color="#F3F3F3", zorder=0)
    b.axvline(0.5, color="black", lw=0.6)
    b.axvline(0.60, color=RED, lw=0.6, ls="--")
    b.set_xlim(0.3, 1.0)
    b.set_xlabel("Verdict AUROC for the event (95% CI)", fontsize=7.5)
    b.tick_params(labelsize=6)
    if not path.exists():
        b.text(0.65, 0.5, "Pending registered runs", transform=b.transAxes, ha="center", fontsize=8.0, color=GREY)
        b.set_yticks([])
    else:
        res = json.load(open(path))["imv"]
        g2 = _gate2_rows(res)
        for i, (label, est, lo, hi, status) in enumerate(g2):
            color = BLUE if status == "control" else (GREEN if status == "open" else RED if status == "closed" else GREY)
            b.errorbar(est, i, xerr=[[est - lo], [hi - est]], fmt="o", color=color, ms=4, capsize=2, lw=1)
        b.set_yticks(range(len(g2)), [r[0] for r in g2], fontsize=7.5)
        b.set_ylim(len(g2) - 0.5, -1.0)  # inverted, with headroom for reference labels
        ref = next(v["LOW"] for v in res.values() if isinstance(v, dict) and "LOW" in v)
        for key, label, ls, dx, ha in (("benchmark_pA", "model A", ":", -0.006, "right"),
                                       ("benchmark_pB", "model B", "-.", 0.006, "left")):
            x = ref[key]["est"]
            b.axvline(x, color=GREY, lw=0.8, ls=ls)
            b.text(x + dx, -0.65, f"{label} {x:.2f}", fontsize=6.5, color=GREY, ha=ha, va="center")
        _capacity_panel(c, g2)
    fig.savefig(OUT / "fig2_two_ceilings_evidence.png", bbox_inches="tight", dpi=600)
    plt.close(fig)


def _capacity_panel(c: "plt.Axes", g2: List[Tuple[str, float, float, float, str]]) -> None:
    """Fig. 2c: observed verdict AUROC against the maximum attainable from the verdict counts."""
    ph = json.load(open(RES / "v4" / "posthoc_v4_gate2.json"))["explainers"]
    keys = {label: key for key, label in EXPLAINERS}
    c.set_title("c  Readout capacity", loc="left", fontsize=8.5, weight="bold")
    c.axvline(0.5, color="black", lw=0.6)
    c.axvline(0.60, color=RED, lw=0.6, ls="--")
    c.text(0.61, -0.65, "0.60 threshold", fontsize=6.5, color=RED, ha="left", va="center")
    for i, (label, est, _, _, status) in enumerate(g2):
        if label not in keys:
            continue
        mx = ph[keys[label]]["taxai"]["max_attainable_auroc"]
        c.plot([est, mx], [i, i], color=GREY, lw=1.0, zorder=1)
        c.plot(mx, i, marker="|", color="black", ms=7, mew=1.2, zorder=2)
        c.plot(est, i, "o", color=RED, ms=4, zorder=3)
        c.text(mx + 0.015, i, f"{mx:.2f}", fontsize=6.5, va="center")
    c.set_xlim(0.45, 0.95)
    c.tick_params(axis="y", labelleft=False)
    c.set_xlabel("AUROC: observed (dot), maximum (bar)", fontsize=7.5)
    c.tick_params(labelsize=6)


def figure3() -> None:
    """Failure modes (H4): verdict composition by outcome, ventilation LOW, per explainer."""
    path = RES / "v4" / "h4_verdict_counts.json"
    if not path.exists():
        logger.warning(f"{path} missing; Figure 3 skipped")
        return
    counts = json.load(open(path))
    rows = [("qwen2.5:14b", "taxai", "qwen2.5:14b"), ("qwen2.5:14b", "taxai_hint", "qwen2.5:14b + hint"),
            ("gemma3:27b", "taxai", "gemma3:27b"), ("qwen2.5:32b", "taxai", "qwen2.5:32b"),
            ("gpt-oss:20b", "taxai", "gpt-oss:20b"), ("qwen2.5:72b-instruct-q3_K_M", "taxai", "qwen2.5:72b")]
    cats = (("too low", GREEN, "too low (would rank ventilated patients higher)"), ("about right", "#D9D9D9", "about right"),
            ("too high", RED, "too high"), ("missing", "white", "missing"))
    g2 = json.load(open(RES / "v4" / "gate2_v4.json"))["imv"]
    fig, ax = plt.subplots(figsize=(7.2, 3.0), dpi=300)
    ylabels, y = [], 0.0
    for model, cond, label in rows:
        auc = g2[model]["LOW"][cond]["auroc"]["est"]
        label = f"{label} (AUROC {auc:.2f})"
        c = counts[f"{model}|LOW|{cond}"]
        for outcome, tag in (("1", "ventilated"), ("0", "not ventilated")):
            n = sum(c[outcome].values())
            left = 0.0
            for key, color, _ in cats:
                w = c[outcome].get(key, 0) / n
                ax.barh(y, w, left=left, color=color, edgecolor="black", lw=0.3, height=0.8)
                left += w
            k = c[outcome].get("too low", 0)
            ax.text(1.01, y, f"{k}/{n}", va="center", fontsize=6.5, color=GREEN if outcome == "1" else GREY)
            ylabels.append((y, f"{label} \u2014 {tag}" if outcome == "1" else tag))
            y += 0.9
        y += 0.5
    ax.set_yticks([t for t, _ in ylabels], [lab for _, lab in ylabels], fontsize=6.8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of verdicts (TAXAI condition; ventilation, LOW labels)", fontsize=7.5)
    ax.tick_params(axis="x", labelsize=6)
    ax.text(1.01, -0.9, '"too low"', fontsize=6.5, color=GREY)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=col, ec="black", lw=0.3) for _, col, _ in cats]
    ax.legend(handles, [lab for _, _, lab in cats], fontsize=6.8, ncol=4, frameon=False,
              loc="upper center", bbox_to_anchor=(0.5, 1.12))
    fig.savefig(OUT / "fig3_failure_modes.png", bbox_inches="tight", dpi=600)
    plt.close(fig)


def figure4() -> None:
    """Pre-specified addendum: readouts and anchoring (ventilation, LOW labels)."""
    path = RES / "v4" / "tierC_v4.json"
    if not path.exists():
        logger.warning(f"{path} missing; Figure 4 skipped")
        return
    tc = json.load(open(path))
    g2 = json.load(open(RES / "v4" / "gate2_v4.json"))["imv"]
    ref = g2["qwen2.5:14b"]["LOW"]
    pa, pb = ref["benchmark_pA"]["est"], ref["benchmark_pB"]["est"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.8), dpi=300, gridspec_kw={"width_ratios": [1.25, 1]})
    readouts = (("verdict", "Verdict (registered)", RED, "o"),
                ("C1a", "Probability, prediction shown", BLUE, "s"),
                ("C1b", "Probability, prediction hidden", GREEN, "D"))
    for i, (key, label) in enumerate(EXPLAINERS):
        for j, (r, _, color, marker) in enumerate(readouts):
            if r == "verdict":
                au = g2[key]["LOW"]["taxai"]["auroc"]
            elif r in tc[key]:
                au = tc[key][r]["parsed_cases"]["auroc"]
            else:
                continue
            y = i + (j - 1) * 0.22
            a.errorbar(au["est"], y, xerr=[[au["est"] - au["ci95"][0]], [au["ci95"][1] - au["est"]]],
                       fmt=marker, color=color, ms=3.5, capsize=1.5, lw=0.9)
    a.axvline(0.5, color="black", lw=0.6)
    for x, label, ls in ((pa, "model A", ":"), (pb, "model B", "-.")):
        a.axvline(x, color=GREY, lw=0.8, ls=ls)
        dx, ha = (-0.005, "right") if label == "model A" else (0.005, "left")
        a.text(x + dx, -0.7, f"{label} {x:.2f}", fontsize=6.5, color=GREY, ha=ha, va="center")
    a.set_yticks(range(len(EXPLAINERS)), [lab for _, lab in EXPLAINERS], fontsize=7.5)
    a.set_ylim(len(EXPLAINERS) - 0.5, -1.0)
    a.set_xlim(0.35, 0.95)
    a.set_xlabel("AUROC for the event (95% CI)", fontsize=7.5)
    a.tick_params(labelsize=6)
    a.set_title("a  Readouts", loc="left", fontsize=8.5, weight="bold")
    handles = [plt.Line2D([], [], color=c, marker=m, ls="", ms=4) for _, _, c, m in readouts]
    fig.legend(handles, [lab for _, lab, _, _ in readouts], fontsize=7, frameon=False, ncol=3,
               loc="upper center", bbox_to_anchor=(0.5, 1.07))

    for key, label in EXPLAINERS:
        for r, color, marker in (("C1a", BLUE, "s"), ("C1b", GREEN, "D")):
            if r not in tc[key]:
                continue
            pc = tc[key][r]["parsed_cases"]
            b.plot(pc["spearman_with_pA"], pc["auroc"]["est"], marker=marker, color=color, ms=4, ls="")
            if r == "C1a":
                b.annotate(label, (pc["spearman_with_pA"], pc["auroc"]["est"]), fontsize=6.5,
                           xytext=(4, -2), textcoords="offset points")
    b.axhline(pa, color=GREY, lw=0.8, ls=":")
    b.text(-0.28, pa + 0.004, f"model A {pa:.2f}", fontsize=6.5, color=GREY, va="bottom")
    b.axhline(0.5, color="black", lw=0.6)
    b.set_xlim(-0.3, 1.3)
    b.set_ylim(0.45, 0.8)
    b.set_xlabel("Spearman correlation with model A's probability", fontsize=7.5)
    b.set_ylabel("AUROC", fontsize=7.5)
    b.tick_params(labelsize=6)
    b.set_title("b  Anchoring on the prediction", loc="left", fontsize=8.5, weight="bold")
    fig.subplots_adjust(wspace=0.3)
    fig.savefig(OUT / "fig4_addendum_readouts.png", bbox_inches="tight", dpi=600)
    plt.close(fig)


def main() -> None:
    figure1()
    figure2()
    figure3()
    figure4()
    logger.info(f"Figures written to {OUT}/")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()

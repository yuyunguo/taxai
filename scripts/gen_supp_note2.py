"""Supplementary Note 2 tables (Gate 2, full results by explainer).

Reads results_v2/v4/gate2_v4.json (aggregate statistics only) and writes
manuscript/supp_note2_tables.tex, which supplement_v4.tex includes. Every
number comes from the registered analysis output; nothing is typed by hand.

Usage:
    python scripts/gen_supp_note2.py
"""

import json
import logging
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)
RES = Path("results_v2/v4/gate2_v4.json")
OUT = Path("manuscript/supp_note2_tables.tex")
EXPLAINERS = (("qwen2.5:14b", "qwen2.5:14b"), ("qwen2.5:32b", "qwen2.5:32b"),
              ("qwen2.5:72b-instruct-q3_K_M", "qwen2.5:72b"), ("gemma3:27b", "gemma3:27b"),
              ("gpt-oss:20b", "gpt-oss:20b"))
CONDS = (("static_llm", "Static LLM"), ("taxai", "TAXAI"), ("taxai_hint", "TAXAI + hint"))


def ci(a: Dict[str, object], key: str = "ci95") -> str:
    lo, hi = a[key]
    return f"{a['est']:.3f} [{lo:.3f}, {hi:.3f}]"


def diff(a: Dict[str, object]) -> str:
    lo, hi = a["ci95"]
    return f"${a['est']:+.3f}$ [${lo:+.3f}$, ${hi:+.3f}$]"


def pct(x: float) -> str:
    return f"{100 * x:.1f}\\%"


def table(caption: str, label: str, cols: str, header: str, rows: List[str], small: bool = True) -> str:
    body = "\n".join(rows)
    size = "\\footnotesize\n" if small else ""
    return (f"\\begin{{table}}[h]\n\\centering\n\\caption{{{caption}}}\n\\label{{{label}}}\n{size}"
            f"\\resizebox{{\\ifdim\\width>\\textwidth\\textwidth\\else\\width\\fi}}{{!}}{{%\n\\begin{{tabular}}{{{cols}}}\n\\toprule\n{header} \\\\\n\\midrule\n"
            f"{body}\n\\bottomrule\n\\end{{tabular}}}}\n\\end{{table}}\n")


def main() -> None:
    res = json.load(open(RES))
    imv, mort = res["imv"], res["mortality"]

    # S1: ventilation LOW discrimination; S1b: verdict behavior.
    rows, rows_b = [], []
    for key, name in EXPLAINERS:
        r = imv[key]["LOW"]
        excl = imv[key]["LOW_excluding_prior_sample"]
        status = imv[key]["gate2_status_taxai"].capitalize()
        for c, cname in CONDS:
            if c not in r:
                continue
            a = r[c]
            h1 = status if c == "taxai" else "---"
            rows.append(f"{name} & {cname} & {ci(a['auroc'])} & {ci(a['auroc'], 'ci99')} & {excl[c]['est']:.3f} & {h1} \\\\")
            inc = r.get("incremental_taxai", {}) if c == "taxai" else {}
            inc_s = f"${inc['coef']:+.2f}$ ($p = {inc['p']:.3f}$)" if "coef" in inc else "---"
            rows_b.append(f"{name} & {cname} & {pct(a['about_right_share'])} & ${a['mean_score_events']:+.3f}$ & "
                          f"${a['mean_score_nonevents']:+.3f}$ & {pct(a['missing_rate'])} & {inc_s} \\\\")
    s1 = table("\\textbf{Explainer ceiling for ventilation, LOW labels: discrimination (registered primary stratum; "
               "$n = 400$, 200 events).} Verdict AUROC with bootstrap 95\\% and Bonferroni-adjusted 99\\% confidence "
               "intervals, and AUROC after excluding the 75 cases from the earlier exploratory sample. H1 is classified on "
               "the TAXAI condition: closed if the upper 95\\% limit is below 0.60.",
               "tab:s_imv_low", "@{}llcccc@{}",
               "\\textbf{Explainer} & \\textbf{Condition} & \\textbf{AUROC (95\\% CI)} & \\textbf{99\\% CI} & "
               "\\textbf{Excl.\\ overlap} & \\textbf{H1}", rows)
    s1b = table("\\textbf{Explainer ceiling for ventilation, LOW labels: verdict behavior (H4).} Share of ``about right''; "
                "mean verdict score ($-1$ too high, 0 about right, $+1$ too low) among events and among non-events; share of "
                "missing verdicts (scored 0); and, for TAXAI, the coefficient of the verdict score in logistic regression of "
                "the outcome on logit(model A) and the verdict.",
                "tab:s_imv_behavior", "@{}llccccc@{}",
                "\\textbf{Explainer} & \\textbf{Condition} & \\textbf{About right} & \\textbf{Mean score, events} & "
                "\\textbf{Mean score, non-events} & \\textbf{Missing} & \\textbf{Incremental value}", rows_b)

    # S2: paired contrasts and references.
    rows = []
    for key, name in EXPLAINERS:
        r = imv[key]["LOW"]
        if "taxai_minus_static" in r:
            rows.append(f"TAXAI $-$ Static LLM & {name} & {diff(r['taxai_minus_static'])} \\\\")
    rows.append(f"TAXAI + hint $-$ TAXAI (H3) & qwen2.5:14b & {diff(imv['qwen2.5:14b']['LOW']['hint_minus_taxai'])} \\\\")
    for k, v in imv["scale_ladder_LOW"].items():
        a, b = k.split("_minus_")
        rows.append(f"Scale ladder & {a} $-$ {b} & {diff(v)} \\\\")
    s2 = table("\\textbf{Paired contrasts, ventilation LOW labels.} Paired bootstrap 95\\% confidence intervals.",
               "tab:s_contrasts", "@{}llc@{}", "\\textbf{Contrast} & \\textbf{Explainer} & \\textbf{Difference in AUROC (95\\% CI)}",
               rows, small=False)

    # S3: references (rule control, models A and B by split).
    low = imv["qwen2.5:14b"]["LOW"]
    rows = [f"Rule control, sign($p_B - p_A$) (H2) & {ci(imv['rule_control_LOW'])} & --- & --- \\\\",
            f"Model A probability & {ci(low['benchmark_pA'])} & {low['benchmarks_val']['pA']['est']:.3f} & {low['benchmarks_test']['pA']['est']:.3f} \\\\",
            f"Model B probability & {ci(low['benchmark_pB'])} & {low['benchmarks_val']['pB']['est']:.3f} & {low['benchmarks_test']['pB']['est']:.3f} \\\\"]
    s3 = table("\\textbf{Reference discrimination, ventilation LOW labels.} AUROC in all 400 cases and separately in the "
               "validation and test cases. Validation probabilities are slightly optimistic because validation data were used "
               "for early stopping.", "tab:s_refs", "@{}lccc@{}",
               "\\textbf{Reference} & \\textbf{All (95\\% CI)} & \\textbf{Validation} & \\textbf{Test}", rows, small=False)

    # S4: ventilation HIGH (descriptive) and S5: mortality by stratum.
    def strat_rows(block: Dict[str, object], strata: List[str]) -> List[str]:
        out = []
        for key, name in EXPLAINERS:
            m = block.get(key)
            if not isinstance(m, dict):
                continue
            for st in strata:
                r = m.get(st)
                if not r:
                    continue
                for c, cname in CONDS:
                    if c in r:
                        out.append(f"{name} & {st} & {cname} & {r['n']} ({r['events']}) & {ci(r[c]['auroc'])} & "
                                   f"{pct(r[c]['about_right_share'])} \\\\")
        return out

    hdr = ("\\textbf{Explainer} & \\textbf{Stratum} & \\textbf{Condition} & \\textbf{$n$ (events)} & "
           "\\textbf{AUROC (95\\% CI)} & \\textbf{About right}")
    s4 = table("\\textbf{Ventilation, HIGH labels (descriptive; 58 events, 58 non-events).} The 72B explainer was not run "
               "in this stratum by design. Rule control: " + ci(imv["qwen2.5:14b"]["HIGH"]["rule_control"]) + ".",
               "tab:s_imv_high", "@{}lllccc@{}", hdr, strat_rows(imv, ["HIGH"]))
    mrule = next(v for v in mort.values() if isinstance(v, dict))
    s5 = table("\\textbf{Mortality control task (secondary; 232 patients).} Verdict AUROC for death within each label "
               "stratum. Rule control: HIGH " + ci(mrule["HIGH"]["rule_control"]) + ", LOW " + ci(mrule["LOW"]["rule_control"]) +
               ". qwen2.5:14b and qwen2.5:32b results on this sample were seen before registration. The LOW stratum (46 "
               "patients) is too small for inference.", "tab:s_mort", "@{}lllccc@{}", hdr, strat_rows(mort, ["HIGH", "LOW"]))

    intro = ("All values come from the registered analysis output (\\texttt{gate2\\_v4\\_analysis.py}); the tables are "
             "generated by script. Supplementary Tables~\\ref{tab:s_imv_low}--\\ref{tab:s_refs} cover the registered "
             "primary stratum; Supplementary Tables~\\ref{tab:s_imv_high} and \\ref{tab:s_mort} are descriptive or "
             "secondary.\n\n")
    OUT.write_text("% Generated by scripts/gen_supp_note2.py from results_v2/v4/gate2_v4.json. Do not edit by hand.\n"
                   + intro + "\n".join([s1, s1b, s2, s3, s4, s5]))
    logger.info(f"Wrote {OUT}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()

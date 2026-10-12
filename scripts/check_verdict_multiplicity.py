"""Audit verdict-parse multiplicity in the v4 generation caches (read-only).

The registered verdict parser (src.pipeline_v2.timeline_measures_v3.VERDICT,
consumed via re.search) scores the FIRST occurrence of
"appears (too high|about right|too low)" in a response. The instruction places
the verdict sentence last, so first-match and last-match parsing coincide
unless the explanation body contains an earlier verdict phrase (for example,
"the lactate appears too high"). This script counts those cases so the
assumption can be verified against the caches on the database host.

Run on the host that holds the caches:
    python scripts/check_verdict_multiplicity.py [cache_dir]

Cache dir defaults to the pipeline's v4 output directory (DATA_DIR / "v4").
Reports, per cache file and condition: responses parsed, responses with more
than one regex match, and responses where first-match and last-match verdicts
differ (the only cases that could change a reported verdict score).
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

VERDICT = re.compile(r"appears (too high|about right|too low)", re.I)
SCORE = {"too high": -1, "about right": 0, "too low": 1}


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else default_dir()
    files = sorted(root.glob("gen_*.jsonl"))
    if not files:
        sys.exit(f"no gen_*.jsonl caches under {root}")
    total_multi = total_diff = total_resp = 0
    for f in files:
        per_cond: Counter = Counter()
        n_resp = n_multi = n_diff = 0
        diffs = []
        for line in open(f):
            r = json.loads(line)
            ms = VERDICT.findall(r["response"] or "")
            n_resp += 1
            if len(ms) > 1:
                n_multi += 1
                first, last = ms[0].lower(), ms[-1].lower()
                per_cond[(r["method"], "multi")] += 1
                if SCORE[first] != SCORE[last]:
                    n_diff += 1
                    diffs.append((r["method"], int(r["hadm_id"]), first, last))
            elif len(ms) == 1:
                per_cond[(r["method"], "parsed")] += 1
            else:
                per_cond[(r["method"], "missing")] += 1
        total_multi += n_multi
        total_diff += n_diff
        total_resp += n_resp
        flag = "  <-- SCORE WOULD CHANGE" if n_diff else ""
        print(f"{f.name}: responses={n_resp} multi_match={n_multi} first!=last={n_diff}{flag}")
        for (method, kind), n in sorted(per_cond.items()):
            print(f"    {method:12s} {kind}: {n}")
        for d in diffs[:20]:
            print(f"    DIFF method={d[0]} hadm_id={d[1]} first={d[2]!r} last={d[3]!r}")
    print(f"\nTOTAL: responses={total_resp} multi_match={total_multi} first!=last={total_diff}")
    print("If first!=last is 0 everywhere, the first-match parser is equivalent to "
          "parsing the final verdict sentence and no reported number can change.")


def default_dir() -> Path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.pipeline_v2.gate2_v4 import OUT
    return OUT


if __name__ == "__main__":
    main()

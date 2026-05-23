#!/usr/bin/env python3
"""collate_judgments.py -- join the two judge passes into one wide table.

Reads (read-only) the two semantic-equivalence judge files:
  - llm_judge_results.jsonl        primary Gemini judge
  - llm_judge_results_llama.jsonl  independent Llama-3.3-70B judge

and emits results/composite/judgments_joined.csv with the two judges' verdicts
side by side per (evaluated_model, pair_index), plus an agreement flag and the
shared metrics. This is the input to compute_interjudge_kappa.py.

Dedupe: keep the last VALID judgment per (evaluated_model, pair_index); stale
or errored append-only attempts are dropped (a successful retry supersedes).
"""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COMP = ROOT / "results" / "composite"
VALID = ("equivalent", "partially_equivalent", "not_equivalent")

GEMINI = COMP / "llm_judge_results.jsonl"
LLAMA = COMP / "llm_judge_results_llama.jsonl"
OUT_CSV = COMP / "judgments_joined.csv"


def load_valid(path):
    """(evaluated_model, pair_index) -> last VALID judge record. Read-only."""
    out = {}
    if not path.exists():
        return out
    for ln in path.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        try:
            r = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if r.get("judgment") in VALID:
            out[(r["evaluated_model"], r["pair_index"])] = r  # last valid wins
    return out


def main():
    gemini = load_valid(GEMINI)
    llama = load_valid(LLAMA)

    keys = sorted(set(gemini) | set(llama))
    cols = ["evaluated_model", "pair_index", "direction",
            "gemini_judgment", "gemini_reason",
            "llama_judgment", "llama_reason",
            "judges_agree", "both_judged",
            "chrf_pp", "bertscore_f1",
            "source_text", "reference_text", "candidate_text"]

    agree = both = 0
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for k in keys:
            m, pi = k
            grec = gemini.get(k)
            lrec = llama.get(k)
            base = grec or lrec  # any present record carries shared metrics
            gj = grec["judgment"] if grec else None
            lj = lrec["judgment"] if lrec else None
            bo = bool(grec and lrec)
            ag = bool(bo and gj == lj)
            both += bo
            agree += ag
            w.writerow({
                "evaluated_model": m, "pair_index": pi,
                "direction": (base or {}).get("direction", "lak_to_eng"),
                "gemini_judgment": gj,
                "gemini_reason": grec["reason"] if grec else None,
                "llama_judgment": lj,
                "llama_reason": lrec["reason"] if lrec else None,
                "judges_agree": ag if bo else "",
                "both_judged": bo,
                "chrf_pp": (base or {}).get("chrf_pp"),
                "bertscore_f1": (base or {}).get("bertscore_f1"),
                "source_text": (base or {}).get("source_text", ""),
                "reference_text": (base or {}).get("reference_text", ""),
                "candidate_text": (base or {}).get("candidate_text", ""),
            })

    print("== SOURCES (read-only, deduped to valid) ==")
    print(f"  Gemini judge : {len(gemini):>5}  models={len({m for m, _ in gemini})}")
    print(f"  Llama judge  : {len(llama):>5}  models={len({m for m, _ in llama})}")
    print("== JOINED ==")
    print(f"  {OUT_CSV.name}: {len(keys)} (model,pair) rows")
    print(f"  both judged (kappa-eligible): {both}")
    if both:
        print(f"  raw agreement on overlap    : {agree}/{both} = {agree / both:.3f}")
    else:
        print("  no overlap yet -- run both judges first")


if __name__ == "__main__":
    main()

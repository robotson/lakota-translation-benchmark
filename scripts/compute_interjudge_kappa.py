#!/usr/bin/env python3
"""compute_interjudge_kappa.py -- inter-judge agreement (Gemini vs Llama).

Reads results/composite/judgments_joined.csv (from collate_judgments.py),
restricts to rows judged by BOTH judges, and reports:
  - raw agreement
  - Cohen's kappa (unweighted, nominal)
  - quadratic-weighted kappa (the 3 labels are ordinal:
      not_equivalent < partially_equivalent < equivalent)
  - 3x3 confusion matrix (Gemini = rows, Llama = cols)
  - per-model kappa, and a proprietary-vs-open-weight split

Pure stdlib (no sklearn) for dependency-stable reproducibility.
"""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "results" / "composite" / "judgments_joined.csv"
ORDER = ["not_equivalent", "partially_equivalent", "equivalent"]
IDX = {c: i for i, c in enumerate(ORDER)}

# Open-weight models are identified by name substring, so the split holds
# regardless of the condition suffix (e.g. deepseek_thinking, qwen_nonthinking).
OSS_KEYS = ("deepseek", "qwen", "glm")


def is_oss(model: str) -> bool:
    return any(k in model.lower() for k in OSS_KEYS)


def kappas(pairs):
    """pairs: list of (gemini_label, llama_label). Returns stats dict or None."""
    n = len(pairs)
    if n == 0:
        return None
    k = len(ORDER)
    cm = [[0] * k for _ in range(k)]
    for g, l in pairs:
        cm[IDX[g]][IDX[l]] += 1
    obs = sum(cm[i][i] for i in range(k)) / n
    row = [sum(cm[i]) / n for i in range(k)]
    col = [sum(cm[r][c] for r in range(k)) / n for c in range(k)]
    exp = sum(row[i] * col[i] for i in range(k))
    kappa = (obs - exp) / (1 - exp) if (1 - exp) else 0.0
    num = den = 0.0
    for i in range(k):
        for j in range(k):
            w = ((i - j) ** 2) / ((k - 1) ** 2)
            num += w * cm[i][j] / n
            den += w * row[i] * col[j]
    wkappa = 1 - num / den if den else 0.0
    return {"n": n, "raw": obs, "kappa": kappa, "wkappa": wkappa, "cm": cm}


def band(x):
    return ("poor" if x < 0 else "slight" if x < .2 else "fair" if x < .4
            else "moderate" if x < .6 else "substantial" if x < .8
            else "almost perfect")


def main():
    if not CSV_PATH.exists():
        raise SystemExit(f"{CSV_PATH} not found -- run collate_judgments.py first")
    rows = list(csv.DictReader(open(CSV_PATH, encoding="utf-8")))
    both = [r for r in rows
            if r["gemini_judgment"] in IDX and r["llama_judgment"] in IDX]
    pairs = [(r["gemini_judgment"], r["llama_judgment"]) for r in both]
    overall = kappas(pairs)
    if not overall:
        raise SystemExit("no rows judged by both judges yet")

    print("== INTER-JUDGE: Gemini-3-flash vs Llama-3.3-70B ==")
    print(f"both-judged pairs: {overall['n']} of {len(rows)} collated")
    print(f"raw agreement    : {overall['raw']:.3f}")
    print(f"Cohen's kappa    : {overall['kappa']:.3f}  ({band(overall['kappa'])})")
    print(f"quad-weighted    : {overall['wkappa']:.3f}  ({band(overall['wkappa'])})")
    print(f"\nconfusion matrix (rows=Gemini, cols=Llama; order={ORDER}):")
    print("                 " + "".join(f"{c[:7]:>9}" for c in ORDER))
    for i, c in enumerate(ORDER):
        print(f"  G:{c:<14}" + "".join(f"{overall['cm'][i][j]:>9}" for j in range(len(ORDER))))

    for label, sel in (("PROPRIETARY", lambda m: not is_oss(m)),
                       ("OPEN-WEIGHT", is_oss)):
        sub = [(r["gemini_judgment"], r["llama_judgment"]) for r in both if sel(r["evaluated_model"])]
        s = kappas(sub)
        if s:
            print(f"\n{label}: n={s['n']} raw={s['raw']:.3f} "
                  f"kappa={s['kappa']:.3f} ({band(s['kappa'])}) wkappa={s['wkappa']:.3f}")

    print("\n== PER-MODEL kappa ==")
    for m in sorted({r["evaluated_model"] for r in both}):
        sub = [(r["gemini_judgment"], r["llama_judgment"]) for r in both
               if r["evaluated_model"] == m]
        s = kappas(sub)
        tag = "  [OSS]" if is_oss(m) else ""
        print(f"  {m:<22} n={s['n']:>4} raw={s['raw']:.3f} k={s['kappa']:.3f} ({band(s['kappa'])}){tag}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
analyze.py -- Compare baseline and thinking evaluation results.

Auto-discovers run directories under results/baseline/raw/ and
results/thinking/raw/, extracts model names from directory names, and
uses the latest run per model+direction. Produces comparison tables
and a summary CSV.

Usage:
    python scripts/analyze.py                                # auto-discover
    python scripts/analyze.py --csv-out out.csv              # custom CSV path
    python scripts/analyze.py --baseline-dir results/baseline/raw

Expects JSONL files named lak_to_eng.jsonl and eng_to_lak.jsonl in each
run directory.
"""

import argparse
import csv
import json
import re
import sys
import statistics
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DIRECTIONS = ["lak_to_eng", "eng_to_lak"]
DIR_LABELS = {"lak_to_eng": "L→E", "eng_to_lak": "E→L"}

# Model name normalization: map directory prefixes to canonical names.
# Directory names follow patterns like:
#   anthropic_claude-opus-4-6_20260228_210838
#   claude-opus-4-6_20260302_065329
#   openai_gpt-5.2_20260228_164804
#   gemini_gemini-3.1-pro-preview_20260302_065329
MODEL_ALIASES = {
    "claude-opus-4-6": "claude-opus-4-6",
    "claude-sonnet-4-6": "claude-sonnet-4-6",
    "gpt-5.2": "gpt-5.2",
    "gemini-3.1-pro-preview": "gemini-3.1-pro",
    "gemini-3.1-pro": "gemini-3.1-pro",
}


def extract_model_name(dirname: str) -> str | None:
    """Extract canonical model name from a run directory name.

    Handles formats:
        {provider}_{model}_{YYYYMMDD}_{HHMMSS}
        {model}_{YYYYMMDD}_{HHMMSS}
    """
    # Strip trailing timestamp: _YYYYMMDD_HHMMSS
    stripped = re.sub(r"_\d{8}_\d{6}$", "", dirname)

    # Try stripping known provider prefixes
    for prefix in ("anthropic_", "openai_", "gemini_"):
        if stripped.startswith(prefix):
            stripped = stripped[len(prefix):]
            break

    # Look up in aliases
    if stripped in MODEL_ALIASES:
        return MODEL_ALIASES[stripped]

    # Try partial match (for future models)
    for key, canonical in MODEL_ALIASES.items():
        if key in stripped:
            return canonical

    return stripped  # return as-is if no alias found


def discover_runs(raw_dir: Path) -> dict[tuple[str, str], Path]:
    """Discover the latest run directory for each (model, direction) pair.

    Scans raw_dir for subdirectories containing {direction}.jsonl files.
    For each model+direction, keeps the directory with the latest timestamp.

    Returns: {(canonical_model, direction): Path}
    """
    if not raw_dir.exists():
        return {}

    # Collect all (model, direction, timestamp, path) tuples
    candidates: list[tuple[str, str, str, Path]] = []

    for subdir in sorted(raw_dir.iterdir()):
        if not subdir.is_dir():
            continue

        model = extract_model_name(subdir.name)
        if model is None:
            continue

        # Extract timestamp for sorting
        ts_match = re.search(r"(\d{8}_\d{6})$", subdir.name)
        timestamp = ts_match.group(1) if ts_match else "00000000_000000"

        for direction in DIRECTIONS:
            jsonl_path = subdir / f"{direction}.jsonl"
            if jsonl_path.exists():
                candidates.append((model, direction, timestamp, subdir))

    # For each (model, direction), keep the latest timestamp
    best: dict[tuple[str, str], tuple[str, Path]] = {}
    for model, direction, timestamp, path in candidates:
        key = (model, direction)
        if key not in best or timestamp > best[key][0]:
            best[key] = (timestamp, path)

    return {key: path for key, (_, path) in best.items()}


def load_results(directory: Path, direction: str) -> list[dict]:
    """Load JSONL results for a given direction."""
    fpath = directory / f"{direction}.jsonl"
    results = []
    with open(fpath) as f:
        for line in f:
            results.append(json.loads(line))
    return results


def classify_result(r: dict) -> str:
    """Classify a result as translation, refusal, empty, or error."""
    if "response_type" in r:
        return r["response_type"]
    if r.get("error"):
        return "error"
    cleaned = r.get("cleaned_response", "")
    if not cleaned or cleaned.strip() == "":
        return "empty"
    if any(phrase in cleaned.lower() for phrase in [
        "i cannot", "i can't", "i'm not able", "i am not able",
        "i don't have", "i do not have", "cannot reliably",
        "not comfortable", "beyond my", "i'm unable"
    ]):
        return "refusal"
    return "translation"


def get_chrf(r: dict) -> float | None:
    for key in ["chrf_pp", "chrf_score"]:
        val = r.get(key)
        if val is not None:
            return float(val)
    return None


def get_norm_chrf(r: dict) -> float | None:
    val = r.get("chrf_pp_normalized")
    if val is not None:
        return float(val)
    return None


def get_bleu(r: dict) -> float | None:
    val = r.get("bleu")
    if val is not None:
        return float(val)
    return None


def get_confidence(r: dict) -> float | None:
    val = r.get("confidence")
    if val is not None:
        return float(val)
    return None


def analyze_run(run_map: dict[tuple[str, str], Path], version: str) -> dict:
    """Analyze a complete run. Returns nested dict: stats[model][direction]."""
    # Group by model
    models = sorted(set(m for m, _ in run_map))
    stats = {}

    for model in models:
        stats[model] = {}
        for direction in DIRECTIONS:
            key = (model, direction)
            if key not in run_map:
                stats[model][direction] = {
                    "n_total": 0, "n_translations": 0, "n_refusals": 0,
                    "n_empties": 0, "n_errors": 0, "chrf_mean": None,
                    "chrf_median": None, "chrf_stdev": None, "chrf_scores": [],
                    "norm_chrf_mean": None, "bleu_mean": None,
                    "conf_mean": None, "conf_stdev": None,
                }
                continue

            results = load_results(run_map[key], direction)

            translations = []
            refusals = 0
            empties = 0
            errors = 0
            chrfs = []
            norm_chrfs = []
            bleus = []
            confs = []

            for r in results:
                rtype = classify_result(r)
                if rtype == "translation":
                    translations.append(r)
                    chrf = get_chrf(r)
                    if chrf is not None:
                        chrfs.append(chrf)
                    nchrf = get_norm_chrf(r)
                    if nchrf is not None:
                        norm_chrfs.append(nchrf)
                    bleu = get_bleu(r)
                    if bleu is not None:
                        bleus.append(bleu)
                    conf = get_confidence(r)
                    if conf is not None:
                        confs.append(conf)
                elif rtype == "refusal":
                    refusals += 1
                elif rtype == "empty":
                    empties += 1
                elif rtype == "error":
                    errors += 1

            n_trans = len(translations)
            stats[model][direction] = {
                "n_total": len(results),
                "n_translations": n_trans,
                "n_refusals": refusals,
                "n_empties": empties,
                "n_errors": errors,
                "chrf_mean": statistics.mean(chrfs) if chrfs else None,
                "chrf_median": statistics.median(chrfs) if chrfs else None,
                "chrf_stdev": statistics.stdev(chrfs) if len(chrfs) > 1 else None,
                "chrf_scores": chrfs,
                "norm_chrf_mean": statistics.mean(norm_chrfs) if norm_chrfs else None,
                "bleu_mean": statistics.mean(bleus) if bleus else None,
                "conf_mean": statistics.mean(confs) if confs else None,
                "conf_stdev": statistics.stdev(confs) if len(confs) > 1 else None,
            }
    return stats


def get_common_models(bl_stats: dict, th_stats: dict) -> list[str]:
    """Return models present in both baseline and thinking, in display order."""
    preferred_order = ["gemini-3.1-pro", "claude-opus-4-6", "claude-sonnet-4-6", "gpt-5.2"]
    common = set(bl_stats.keys()) & set(th_stats.keys())
    ordered = [m for m in preferred_order if m in common]
    ordered += sorted(common - set(ordered))
    return ordered


def print_comparison_table(bl_stats, th_stats):
    models = get_common_models(bl_stats, th_stats)

    print("\n" + "=" * 120)
    print("BASELINE vs THINKING COMPARISON")
    print("Baseline: temp=0, no thinking (except Gemini default) | Thinking: temp=1, reasoning=high")
    print("=" * 120)

    header = f"{'Model':<22} {'Dir':>4}  {'N_bl':>4} {'chrF_bl':>8} {'BLEU_bl':>8}  {'N_th':>4} {'chrF_th':>8} {'Norm_th':>8} {'BLEU_th':>8} {'Conf':>5} {'Ref':>3} {'Emp':>3} {'Err':>3}  {'Δ chrF':>7}"
    print(header)
    print("-" * 120)

    for model in models:
        for direction in DIRECTIONS:
            bl = bl_stats[model][direction]
            th = th_stats[model][direction]

            bl_chrf = bl["chrf_mean"]
            th_chrf = th["chrf_mean"]
            th_norm = th["norm_chrf_mean"]

            delta = ""
            if bl_chrf is not None and th_chrf is not None:
                d = th_chrf - bl_chrf
                delta = f"{d:+.1f}"

            bl_chrf_s = f"{bl_chrf:.1f}" if bl_chrf else "—"
            th_chrf_s = f"{th_chrf:.1f}" if th_chrf else "—"
            th_norm_s = f"{th_norm:.1f}" if th_norm else "—"
            bl_bleu_s = f"{bl['bleu_mean']:.1f}" if bl["bleu_mean"] else "—"
            th_bleu_s = f"{th['bleu_mean']:.1f}" if th["bleu_mean"] else "—"
            conf_s = f"{th['conf_mean']:.2f}" if th["conf_mean"] else "—"

            print(f"{model:<22} {DIR_LABELS[direction]:>4}  "
                  f"{bl['n_translations']:>4} {bl_chrf_s:>8} {bl_bleu_s:>8}  "
                  f"{th['n_translations']:>4} {th_chrf_s:>8} {th_norm_s:>8} {th_bleu_s:>8} {conf_s:>5} "
                  f"{th['n_refusals']:>3} {th['n_empties']:>3} {th['n_errors']:>3}  "
                  f"{delta:>7}")
        print()


def print_model_ranking(bl_stats, th_stats):
    models = get_common_models(bl_stats, th_stats)

    print("\n" + "=" * 80)
    print("MODEL RANKINGS BY chrF++ (translations only)")
    print("=" * 80)

    for direction in DIRECTIONS:
        print(f"\n--- {DIR_LABELS[direction]} ---")

        bl_scores = [(m, bl_stats[m][direction]["chrf_mean"]) for m in models
                      if bl_stats[m][direction]["chrf_mean"] is not None]
        bl_scores.sort(key=lambda x: x[1], reverse=True)
        print(f"  Baseline:  ", end="")
        print(" > ".join(f"{m} ({s:.1f})" for m, s in bl_scores))

        th_scores = [(m, th_stats[m][direction]["chrf_mean"]) for m in models
                      if th_stats[m][direction]["chrf_mean"] is not None
                      and th_stats[m][direction]["n_translations"] >= 50]
        th_scores.sort(key=lambda x: x[1], reverse=True)
        print(f"  Thinking:  ", end="")
        print(" > ".join(f"{m} ({s:.1f})" for m, s in th_scores))


def compute_quartiles(scores: list[float]) -> dict:
    if not scores:
        return {}
    s = sorted(scores)
    n = len(s)
    return {
        "min": s[0],
        "q1": s[n // 4],
        "median": statistics.median(s),
        "q3": s[3 * n // 4],
        "max": s[-1],
        "mean": statistics.mean(s),
    }


def print_distribution_table(bl_stats, th_stats):
    models = get_common_models(bl_stats, th_stats)

    print("\n" + "=" * 100)
    print("chrF++ SCORE DISTRIBUTIONS")
    print("=" * 100)

    for direction in DIRECTIONS:
        print(f"\n--- {DIR_LABELS[direction]} ---")
        print(f"  {'Model':<22} {'Mode':>5}  {'Min':>6} {'Q1':>6} {'Med':>6} {'Q3':>6} {'Max':>6} {'Mean':>6} {'σ':>6}")
        print(f"  {'-'*22} {'-'*5}  {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")

        for model in models:
            for label, stats in [("base", bl_stats), ("think", th_stats)]:
                scores = stats[model][direction]["chrf_scores"]
                if len(scores) < 10:
                    continue
                q = compute_quartiles(scores)
                sd = statistics.stdev(scores) if len(scores) > 1 else 0
                print(f"  {model:<22} {label:>5}  {q['min']:>6.1f} {q['q1']:>6.1f} {q['median']:>6.1f} "
                      f"{q['q3']:>6.1f} {q['max']:>6.1f} {q['mean']:>6.1f} {sd:>6.1f}")


def print_refusal_analysis(th_stats):
    models = sorted(th_stats.keys())

    print("\n" + "=" * 80)
    print("THINKING MODE — RESPONSE TYPE ANALYSIS")
    print("=" * 80)

    for model in models:
        total_ref = sum(th_stats[model][d]["n_refusals"] for d in DIRECTIONS)
        total_emp = sum(th_stats[model][d]["n_empties"] for d in DIRECTIONS)
        total_err = sum(th_stats[model][d]["n_errors"] for d in DIRECTIONS)
        if total_ref + total_emp + total_err == 0:
            continue

        print(f"\n  {model}:")
        for d in DIRECTIONS:
            s = th_stats[model][d]
            if s["n_refusals"] + s["n_empties"] + s["n_errors"] > 0:
                print(f"    {DIR_LABELS[d]}: {s['n_refusals']} refusals, "
                      f"{s['n_empties']} empty, {s['n_errors']} errors "
                      f"(of {s['n_total']} total)")


def print_confidence_analysis(th_stats):
    models = sorted(th_stats.keys())

    print("\n" + "=" * 80)
    print("THINKING MODE — CONFIDENCE vs PERFORMANCE")
    print("=" * 80)
    print(f"  {'Model':<22} {'Dir':>4}  {'chrF++':>7} {'Conf μ':>7} {'Conf σ':>7}")
    print(f"  {'-'*22} {'-'*4}  {'-'*7} {'-'*7} {'-'*7}")

    for model in models:
        for d in DIRECTIONS:
            s = th_stats[model][d]
            if s["conf_mean"] is not None:
                chrf_s = f"{s['chrf_mean']:.1f}" if s["chrf_mean"] else "—"
                conf_s = f"{s['conf_mean']:.2f}"
                conf_sd = f"{s['conf_stdev']:.2f}" if s["conf_stdev"] else "—"
                print(f"  {model:<22} {DIR_LABELS[d]:>4}  {chrf_s:>7} {conf_s:>7} {conf_sd:>7}")


def print_thinking_effect(bl_stats, th_stats):
    models = get_common_models(bl_stats, th_stats)
    # Exclude Gemini — thinking was already enabled in baseline
    models = [m for m in models if "gemini" not in m.lower()]

    print("\n" + "=" * 80)
    print("THINKING EFFECT (baseline → thinking)")
    print("Note: Gemini excluded — thinking was already enabled in baseline")
    print("=" * 80)
    print(f"  {'Model':<22} {'Dir':>4}  {'Base':>8} {'Think':>8} {'Δ':>7} {'Δ%':>6}")
    print(f"  {'-'*22} {'-'*4}  {'-'*8} {'-'*8} {'-'*7} {'-'*6}")

    for model in models:
        for d in DIRECTIONS:
            bc = bl_stats[model][d]["chrf_mean"]
            tc = th_stats[model][d]["chrf_mean"]
            if bc and tc:
                delta = tc - bc
                pct = (delta / bc * 100) if bc != 0 else 0
                print(f"  {model:<22} {DIR_LABELS[d]:>4}  {bc:>8.1f} {tc:>8.1f} {delta:>+7.1f} {pct:>+5.1f}%")


def save_csv_summary(bl_stats, th_stats, outpath: Path):
    models = get_common_models(bl_stats, th_stats)

    outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "direction", "mode",
            "n_translations", "n_refusals", "n_empties", "n_errors",
            "chrf_mean", "chrf_median", "chrf_stdev",
            "norm_chrf_mean", "bleu_mean",
            "conf_mean", "conf_stdev",
        ])

        for model in models:
            for d in DIRECTIONS:
                for mode, stats in [("baseline", bl_stats), ("thinking", th_stats)]:
                    s = stats[model][d]
                    writer.writerow([
                        model, DIR_LABELS[d], mode,
                        s["n_translations"], s["n_refusals"], s["n_empties"], s["n_errors"],
                        f"{s['chrf_mean']:.2f}" if s["chrf_mean"] else "",
                        f"{s['chrf_median']:.2f}" if s["chrf_median"] else "",
                        f"{s['chrf_stdev']:.2f}" if s["chrf_stdev"] else "",
                        f"{s['norm_chrf_mean']:.2f}" if s["norm_chrf_mean"] else "",
                        f"{s['bleu_mean']:.2f}" if s["bleu_mean"] else "",
                        f"{s['conf_mean']:.2f}" if s["conf_mean"] else "",
                        f"{s['conf_stdev']:.2f}" if s["conf_stdev"] else "",
                    ])
    print(f"\nCSV saved to {outpath}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare baseline and thinking evaluation results.",
    )
    parser.add_argument("--baseline-dir", type=Path, default=None,
                        help="Baseline raw results directory (default: results/baseline/raw/)")
    parser.add_argument("--thinking-dir", type=Path, default=None,
                        help="Thinking raw results directory (default: results/thinking/raw/)")
    parser.add_argument("--csv-out", type=Path, default=None,
                        help="Output CSV path (default: results/comparison.csv)")
    return parser.parse_args()


def main():
    args = parse_args()

    bl_raw = args.baseline_dir or (PROJECT_ROOT / "results" / "baseline" / "raw")
    th_raw = args.thinking_dir or (PROJECT_ROOT / "results" / "thinking" / "raw")
    csv_out = args.csv_out or (PROJECT_ROOT / "results" / "comparison.csv")

    print(f"Baseline results: {bl_raw}")
    print(f"Thinking results: {th_raw}")

    print("\nDiscovering baseline runs...")
    bl_map = discover_runs(bl_raw)
    if not bl_map:
        print(f"  No baseline results found in {bl_raw}", file=sys.stderr)
        sys.exit(1)
    for (model, direction), path in sorted(bl_map.items()):
        print(f"  {model} {DIR_LABELS[direction]}: {path.name}")

    print("\nDiscovering thinking runs...")
    th_map = discover_runs(th_raw)
    if not th_map:
        print(f"  No thinking results found in {th_raw}", file=sys.stderr)
        sys.exit(1)
    for (model, direction), path in sorted(th_map.items()):
        print(f"  {model} {DIR_LABELS[direction]}: {path.name}")

    print("\nAnalyzing baseline...")
    bl_stats = analyze_run(bl_map, "baseline")

    print("Analyzing thinking...")
    th_stats = analyze_run(th_map, "thinking")

    print_comparison_table(bl_stats, th_stats)
    print_model_ranking(bl_stats, th_stats)
    print_distribution_table(bl_stats, th_stats)
    print_thinking_effect(bl_stats, th_stats)
    print_refusal_analysis(th_stats)
    print_confidence_analysis(th_stats)

    save_csv_summary(bl_stats, th_stats, csv_out)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
run_eval.py -- LLM Translation Evaluation (Lakota ↔ English)

Evaluates LLMs on bidirectional Lakota-English translation using structured
JSON output. Scores with chrF++ (raw and diacritic-normalized) and BLEU.

Two modes:
  Baseline (default):  temperature 0, no extended thinking.
  Thinking (--thinking): temperature 1, extended thinking at max provider
                         settings. Captures reasoning text in output.

Usage:
    python scripts/run_eval.py                              # baseline, full run
    python scripts/run_eval.py --thinking                   # thinking, full run
    python scripts/run_eval.py --thinking --pilot           # 20-pair variance pilot
    python scripts/run_eval.py --dry-run                    # show what would run
    python scripts/run_eval.py --sample 5                   # test on 5 pairs
    python scripts/run_eval.py --models claude-opus-4-6     # single model
    python scripts/run_eval.py --thinking --runs 3          # repeated evaluation
    python scripts/run_eval.py --direction lak_to_eng       # single direction

Requires:
    - API keys in .env (ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY)
    - pip install litellm sacrebleu python-dotenv
"""

import argparse
import csv
import json
import logging
import random
import statistics
import sys
import time
import unicodedata
import uuid
from datetime import datetime, timezone
from pathlib import Path

try:
    import litellm
except ImportError:
    sys.exit("ERROR: litellm not installed. Run: pip install litellm")

try:
    import sacrebleu
except ImportError:
    sys.exit("ERROR: sacrebleu not installed. Run: pip install sacrebleu")

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HOLDOUT_DIR = PROJECT_ROOT / "data" / "holdout"

DEFAULT_MODELS = [
    "openai/gpt-5.2",
    "claude-sonnet-4-6",
    "claude-opus-4-6",
    "gemini/gemini-3.1-pro-preview",
]

RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "translation_result",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "translation": {"type": "string"},
                "confidence": {"type": "number"},
                "refusal_reason": {"type": ["string", "null"]},
            },
            "required": ["translation", "confidence", "refusal_reason"],
            "additionalProperties": False,
        },
    },
}

SYSTEM_PROMPT = (
    "You are a translation system. Translate the input text. "
    "Respond as JSON with three fields: "
    '"translation" (your best translation, or empty string if you cannot translate), '
    '"confidence" (a number from 0.0 to 1.0 representing your confidence in the translation quality), '
    '"refusal_reason" (null if you attempted a translation, or a brief explanation if you cannot translate).'
)

USER_PROMPT_LAK_TO_ENG = "Translate from Lakota to English: {text}"
USER_PROMPT_ENG_TO_LAK = "Translate from English to Lakota: {text}"

# Mode-specific defaults
BASELINE_DEFAULTS = {"temperature": 0, "max_tokens": 1024, "timeout": 60}
THINKING_DEFAULTS = {"temperature": 1, "max_tokens": 16384, "timeout": 180}

DEFAULT_DELAY = 1.5
DEFAULT_MAX_RETRIES = 3
DEFAULT_SEED = 42
DEFAULT_PILOT_PAIRS = 20
DEFAULT_PILOT_RUNS = 3

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"


# ---------------------------------------------------------------------------
# Per-model thinking configuration
# ---------------------------------------------------------------------------

def get_thinking_params(model: str) -> dict:
    """Return provider-specific thinking parameters for a model."""
    if model.startswith("claude-"):
        return {"thinking": {"type": "enabled", "budget_tokens": 8192}}
    elif model.startswith("openai/"):
        return {"reasoning_effort": "high"}
    elif model.startswith("gemini/"):
        return {"reasoning_effort": "high"}
    else:
        return {}


def describe_thinking(model: str) -> str:
    """Human-readable description of thinking config for logging."""
    if model.startswith("claude-"):
        return "extended_thinking(budget_tokens=8192)"
    elif model.startswith("openai/"):
        return "reasoning_effort=high"
    elif model.startswith("gemini/"):
        return "reasoning_effort=high(->thinkingLevel=high)"
    else:
        return "none"


# ---------------------------------------------------------------------------
# Diacritic normalization
# ---------------------------------------------------------------------------

def strip_diacritics(text: str) -> str:
    """Strip all combining marks via NFD decomposition."""
    nfd = unicodedata.normalize("NFD", text)
    stripped = "".join(c for c in nfd if unicodedata.category(c) not in ("Mn", "Mc"))
    return unicodedata.normalize("NFC", stripped)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_holdout_data(holdout_dir: Path) -> list[dict]:
    """Load all holdout JSON files. Returns flat list of pair dicts."""
    all_pairs = []
    json_files = sorted(holdout_dir.glob("*.json"))

    if not json_files:
        logging.error("No JSON files found in %s", holdout_dir)
        sys.exit(1)

    for filepath in json_files:
        logging.info("Loading %s", filepath.name)
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        source_name = data.get("source", filepath.stem)
        register = data.get("register", "unknown")

        for i, pair in enumerate(data.get("pairs", [])):
            all_pairs.append({
                "lakota": pair["lakota"],
                "english": pair["english"],
                "context": pair.get("context", ""),
                "source_file": filepath.name,
                "source_name": source_name,
                "register": register,
                "pair_index": i,
            })

    logging.info("Loaded %d total pairs from %d files", len(all_pairs), len(json_files))
    return all_pairs


# ---------------------------------------------------------------------------
# LLM API calls
# ---------------------------------------------------------------------------

def call_llm_structured(
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    max_retries: int = DEFAULT_MAX_RETRIES,
    thinking: bool = False,
) -> dict:
    """Call LLM with structured JSON output, optionally with thinking enabled."""
    thinking_params = get_thinking_params(model) if thinking else {}
    start_time = time.monotonic()
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                wait = min(2 ** attempt, 30)
                logging.info("    Retry %d/%d after %ds...", attempt, max_retries, wait)
                time.sleep(wait)

            response = litellm.completion(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                response_format=RESPONSE_SCHEMA,
                **thinking_params,
            )

            latency_ms = int((time.monotonic() - start_time) * 1000)
            message = response.choices[0].message
            raw_response = message.content or ""
            finish_reason = response.choices[0].finish_reason

            usage = response.usage if response.usage else None
            input_tokens = usage.prompt_tokens if usage else None
            output_tokens = usage.completion_tokens if usage else None

            # Extract thinking/reasoning content
            thinking_text = None
            if thinking:
                reasoning_content = getattr(message, "reasoning_content", None)
                thinking_blocks = getattr(message, "thinking_blocks", None)
                if reasoning_content:
                    thinking_text = str(reasoning_content)
                elif thinking_blocks:
                    thinking_text = str(thinking_blocks)

            # Parse JSON
            parsed = None
            parse_error = None
            if raw_response:
                try:
                    parsed = json.loads(raw_response)
                except json.JSONDecodeError as e:
                    parse_error = str(e)

            return {
                "raw_response": raw_response,
                "parsed": parsed,
                "parse_error": parse_error,
                "finish_reason": finish_reason,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "thinking_text": thinking_text,
                "latency_ms": latency_ms,
                "error": None,
                "attempts": attempt + 1,
            }

        except Exception as e:
            last_error = e
            if "timeout" in str(e).lower() or "rate" in str(e).lower():
                continue
            else:
                break

    latency_ms = int((time.monotonic() - start_time) * 1000)
    return {
        "raw_response": "",
        "parsed": None,
        "parse_error": None,
        "finish_reason": None,
        "input_tokens": None,
        "output_tokens": None,
        "thinking_text": None,
        "latency_ms": latency_ms,
        "error": str(last_error),
        "attempts": max_retries + 1,
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_chrf_pp(hypothesis: str, reference: str) -> float:
    result = sacrebleu.sentence_chrf(hypothesis, [reference], char_order=6, word_order=2)
    return round(result.score, 2)


def compute_bleu(hypothesis: str, reference: str) -> float:
    result = sacrebleu.sentence_bleu(hypothesis, [reference])
    return round(result.score, 2)


def compute_corpus_metrics(hypotheses: list[str], references: list[str]) -> dict:
    refs_wrapped = [references]
    chrf_result = sacrebleu.corpus_chrf(hypotheses, refs_wrapped, char_order=6, word_order=2)
    bleu_result = sacrebleu.corpus_bleu(hypotheses, refs_wrapped)
    return {
        "chrf_pp": round(chrf_result.score, 2),
        "bleu": round(bleu_result.score, 2),
    }


# ---------------------------------------------------------------------------
# Response classification
# ---------------------------------------------------------------------------

def classify_response(parsed: dict | None, raw: str, error: str | None) -> str:
    if error:
        return "error"
    if parsed is None:
        return "parse_error" if raw else "empty"
    refusal = parsed.get("refusal_reason")
    if refusal is not None and refusal != "":
        return "refusal"
    translation = parsed.get("translation", "")
    if not translation.strip():
        return "empty"
    return "translation"


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_pair(
    pair: dict, model: str, direction: str, delay: float,
    mode: str, temperature: float, max_tokens: int, timeout: int,
    run_id: int = 0, dry_run: bool = False,
) -> dict:
    """Evaluate one pair in one direction."""
    thinking = (mode == "thinking")

    if direction == "lak_to_eng":
        source_text = pair["lakota"]
        reference_text = pair["english"]
        user_prompt = USER_PROMPT_LAK_TO_ENG.format(text=source_text)
    else:
        source_text = pair["english"]
        reference_text = pair["lakota"]
        user_prompt = USER_PROMPT_ENG_TO_LAK.format(text=source_text)

    result = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "run_id": run_id,
        "model": model,
        "direction": direction,
        "source_text": source_text,
        "reference_text": reference_text,
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": user_prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "source_file": pair["source_file"],
        "source_name": pair["source_name"],
        "register": pair["register"],
        "pair_index": pair["pair_index"],
        "context": pair["context"],
    }

    if thinking:
        result["thinking_config"] = describe_thinking(model)

    if dry_run:
        result.update({
            "raw_response": "[DRY RUN]",
            "parse_error": None,
            "finish_reason": None,
            "translation": "",
            "confidence": None,
            "refusal_reason": None,
            "response_type": "dry_run",
            "chrf_pp": None,
            "bleu": None,
            "chrf_pp_normalized": None,
            "bleu_normalized": None,
            "input_tokens": None,
            "output_tokens": None,
            "thinking_text": None,
            "latency_ms": 0,
            "error": None,
            "attempts": 0,
        })
        return result

    api_result = call_llm_structured(
        model, SYSTEM_PROMPT, user_prompt,
        temperature=temperature, max_tokens=max_tokens,
        timeout=timeout, thinking=thinking,
    )
    result.update({
        "raw_response": api_result["raw_response"],
        "parse_error": api_result["parse_error"],
        "finish_reason": api_result["finish_reason"],
        "input_tokens": api_result["input_tokens"],
        "output_tokens": api_result["output_tokens"],
        "thinking_text": api_result["thinking_text"],
        "latency_ms": api_result["latency_ms"],
        "error": api_result["error"],
        "attempts": api_result["attempts"],
    })

    parsed = api_result["parsed"]

    if parsed:
        translation = parsed.get("translation", "").strip()
        confidence = parsed.get("confidence")
        refusal_reason = parsed.get("refusal_reason")
    else:
        translation = ""
        confidence = None
        refusal_reason = None

    result["translation"] = translation
    result["confidence"] = confidence
    result["refusal_reason"] = refusal_reason
    result["response_type"] = classify_response(parsed, api_result["raw_response"], api_result["error"])

    if result["response_type"] == "translation" and translation:
        result["chrf_pp"] = compute_chrf_pp(translation, reference_text)
        result["bleu"] = compute_bleu(translation, reference_text)
        trans_norm = strip_diacritics(translation)
        ref_norm = strip_diacritics(reference_text)
        result["chrf_pp_normalized"] = compute_chrf_pp(trans_norm, ref_norm)
        result["bleu_normalized"] = compute_bleu(trans_norm, ref_norm)
    else:
        result["chrf_pp"] = 0.0
        result["bleu"] = 0.0
        result["chrf_pp_normalized"] = 0.0
        result["bleu_normalized"] = 0.0

    if api_result["parse_error"]:
        result["parsed_raw"] = api_result["raw_response"]

    if delay > 0:
        time.sleep(delay)

    return result


def run_evaluation(
    pairs: list[dict], models: list[str], delay: float,
    mode: str, temperature: float, max_tokens: int, timeout: int,
    num_runs: int = 1, dry_run: bool = False, directions: list[str] | None = None,
) -> list[dict]:
    """Run full evaluation: all models x all pairs x directions x N runs."""
    thinking = (mode == "thinking")
    if directions is None:
        directions = ["lak_to_eng", "eng_to_lak"]
    all_results = []
    total_calls = len(pairs) * len(models) * len(directions) * num_runs
    call_num = 0

    for run_id in range(num_runs):
        if num_runs > 1:
            logging.info("=" * 60)
            logging.info("RUN %d / %d", run_id + 1, num_runs)
            logging.info("=" * 60)

        for model in models:
            logging.info("=" * 60)
            if thinking:
                logging.info("Evaluating model: %s [thinking: %s]", model, describe_thinking(model))
            else:
                logging.info("Evaluating model: %s", model)
            logging.info("=" * 60)

            for direction in directions:
                dir_label = "Lakota -> English" if direction == "lak_to_eng" else "English -> Lakota"
                logging.info("Direction: %s", dir_label)

                for i, pair in enumerate(pairs):
                    call_num += 1
                    run_label = f" run {run_id+1}" if num_runs > 1 else ""
                    logging.info(
                        "  [%d/%d] %s | %s | pair %d/%d%s",
                        call_num, total_calls, model, dir_label, i + 1, len(pairs), run_label,
                    )

                    result = evaluate_pair(
                        pair, model, direction, delay,
                        mode, temperature, max_tokens, timeout,
                        run_id, dry_run,
                    )
                    all_results.append(result)

                    if not dry_run:
                        rtype = result["response_type"]
                        has_thinking = bool(result.get("thinking_text"))
                        think_tag = " [T]" if has_thinking else ""
                        if rtype == "translation":
                            conf = result["confidence"]
                            chrf = result["chrf_pp"]
                            logging.info(
                                "    -> chrF++ %.1f | conf %.2f%s | %s",
                                chrf, conf if conf is not None else -1,
                                think_tag,
                                result["translation"][:60],
                            )
                        elif rtype == "refusal":
                            logging.warning(
                                "    -> REFUSAL%s: %s", think_tag,
                                result["refusal_reason"][:80] if result["refusal_reason"] else "",
                            )
                        elif rtype == "error":
                            logging.warning("    -> ERROR: %s", result["error"][:80] if result["error"] else "")
                        elif rtype == "empty":
                            logging.warning("    -> EMPTY%s (confidence: %s)", think_tag, result["confidence"])
                        else:
                            logging.warning("    -> %s: %s", rtype, result["raw_response"][:80])

    return all_results


# ---------------------------------------------------------------------------
# Pilot variance analysis
# ---------------------------------------------------------------------------

def run_pilot(
    pairs: list[dict], models: list[str], delay: float, num_runs: int,
    mode: str, temperature: float, max_tokens: int, timeout: int,
    results_dir: Path,
) -> None:
    """Run pilot: N runs on subset, report variance per model/direction."""
    logging.info("=" * 60)
    logging.info("PILOT MODE: %d pairs x %d runs x %d models x 2 directions",
                 len(pairs), num_runs, len(models))
    logging.info("Total calls: %d", len(pairs) * num_runs * len(models) * 2)
    logging.info("=" * 60)

    results = run_evaluation(
        pairs, models, delay, mode, temperature, max_tokens, timeout,
        num_runs=num_runs,
    )

    # Group by (model, direction, pair_index) -> list of results across runs
    groups: dict[tuple, list[dict]] = {}
    for r in results:
        if r["response_type"] not in ("translation", "refusal"):
            continue
        key = (r["model"], r["direction"], r["pair_index"])
        groups.setdefault(key, []).append(r)

    # Compute per-pair variance, then aggregate by model/direction
    model_dir_stats: dict[tuple, dict] = {}
    for (model, direction, pair_idx), runs in groups.items():
        md_key = (model, direction)
        if md_key not in model_dir_stats:
            model_dir_stats[md_key] = {
                "chrf_stdevs": [],
                "conf_stdevs": [],
                "chrf_means": [],
                "conf_means": [],
                "n_pairs": 0,
                "refusal_variance": 0,
            }

        stats = model_dir_stats[md_key]
        stats["n_pairs"] += 1

        chrfs = [r["chrf_pp"] for r in runs if r["chrf_pp"] is not None]
        confs = [r["confidence"] for r in runs if r["confidence"] is not None]
        types = [r["response_type"] for r in runs]

        if len(chrfs) >= 2:
            stats["chrf_stdevs"].append(statistics.stdev(chrfs))
            stats["chrf_means"].append(statistics.mean(chrfs))
        if len(confs) >= 2:
            stats["conf_stdevs"].append(statistics.stdev(confs))
            stats["conf_means"].append(statistics.mean(confs))

        if len(set(types)) > 1:
            stats["refusal_variance"] += 1

    # Print report
    print("\n" + "=" * 100)
    print("PILOT VARIANCE ANALYSIS")
    print(f"  {len(pairs)} pairs x {num_runs} runs per model/direction")
    print("=" * 100)
    print(f"\n{'Model':<36} {'Dir':<6} {'Pairs':>5} "
          f"{'chrF++ σ':>9} {'chrF++ μ':>9} "
          f"{'Conf σ':>8} {'Conf μ':>8} "
          f"{'Mixed':>6}")
    print("-" * 100)

    for (model, direction), stats in sorted(model_dir_stats.items()):
        dir_label = "L→E" if direction == "lak_to_eng" else "E→L"

        mean_chrf_sd = (statistics.mean(stats["chrf_stdevs"])
                        if stats["chrf_stdevs"] else 0.0)
        mean_chrf_mu = (statistics.mean(stats["chrf_means"])
                        if stats["chrf_means"] else 0.0)
        mean_conf_sd = (statistics.mean(stats["conf_stdevs"])
                        if stats["conf_stdevs"] else 0.0)
        mean_conf_mu = (statistics.mean(stats["conf_means"])
                        if stats["conf_means"] else 0.0)

        print(f"  {model:<34} {dir_label:<6} {stats['n_pairs']:>5} "
              f"{mean_chrf_sd:>9.2f} {mean_chrf_mu:>9.1f} "
              f"{mean_conf_sd:>8.3f} {mean_conf_mu:>8.2f} "
              f"{stats['refusal_variance']:>6}")

    print("=" * 100)
    print("\nInterpretation:")
    print("  chrF++ σ: mean std dev of chrF++ across runs for the same pair")
    print("            < 3.0 = low variance (single run sufficient)")
    print("            3-8   = moderate (consider multi-run with median)")
    print("            > 8   = high (multi-run recommended)")
    print("  Conf σ:   mean std dev of confidence across runs")
    print("  Mixed:    pairs where response type changed across runs")

    save_raw_results(results, results_dir / "pilot")


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_raw_results(results: list[dict], output_dir: Path) -> None:
    """Save per-instance results as JSONL, organized by model and direction."""
    by_model: dict[str, list[dict]] = {}
    for r in results:
        by_model.setdefault(r["model"], []).append(r)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for model, model_results in by_model.items():
        safe_model = model.replace("/", "_").replace(":", "_")
        model_dir = output_dir / "raw" / f"{safe_model}_{timestamp}"
        model_dir.mkdir(parents=True, exist_ok=True)

        by_direction: dict[str, list[dict]] = {}
        for r in model_results:
            by_direction.setdefault(r["direction"], []).append(r)

        for direction, dir_results in by_direction.items():
            filepath = model_dir / f"{direction}.jsonl"
            with open(filepath, "w", encoding="utf-8") as f:
                for r in dir_results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            logging.info("Saved %d results to %s", len(dir_results), filepath)


def compute_summary(results: list[dict]) -> list[dict]:
    """Compute summary stats grouped by model and direction."""
    groups: dict[tuple, list[dict]] = {}
    for r in results:
        key = (r["model"], r["direction"])
        groups.setdefault(key, []).append(r)

    summary_rows = []

    for (model, direction), group in sorted(groups.items()):
        translations = [r for r in group if r["response_type"] == "translation"]
        refusals = [r for r in group if r["response_type"] == "refusal"]
        errors = [r for r in group if r["response_type"] == "error"]
        empties = [r for r in group if r["response_type"] == "empty"]
        parse_errors = [r for r in group if r["response_type"] == "parse_error"]

        n_total = len(group)
        n_trans = len(translations)

        if translations:
            hyps = [r["translation"] for r in translations]
            refs = [r["reference_text"] for r in translations]
            corpus = compute_corpus_metrics(hyps, refs)

            hyps_norm = [strip_diacritics(h) for h in hyps]
            refs_norm = [strip_diacritics(r) for r in refs]
            corpus_norm = compute_corpus_metrics(hyps_norm, refs_norm)

            confs = [r["confidence"] for r in translations if r["confidence"] is not None]
            mean_conf = round(sum(confs) / len(confs), 3) if confs else None

            has_thinking = sum(1 for r in translations if r.get("thinking_text"))
            thinking_lens = [len(r["thinking_text"]) for r in translations if r.get("thinking_text")]
            mean_thinking_len = round(sum(thinking_lens) / len(thinking_lens)) if thinking_lens else 0
        else:
            corpus = {"chrf_pp": 0.0, "bleu": 0.0}
            corpus_norm = {"chrf_pp": 0.0, "bleu": 0.0}
            mean_conf = None
            has_thinking = 0
            mean_thinking_len = 0

        row = {
            "model": model,
            "direction": direction,
            "n_pairs": n_total,
            "n_translations": n_trans,
            "n_refusals": len(refusals),
            "n_empties": len(empties),
            "n_errors": len(errors),
            "n_parse_errors": len(parse_errors),
            "corpus_chrf_pp": corpus["chrf_pp"],
            "corpus_bleu": corpus["bleu"],
            "corpus_chrf_pp_norm": corpus_norm["chrf_pp"],
            "corpus_bleu_norm": corpus_norm["bleu"],
            "mean_confidence": mean_conf,
        }

        if has_thinking:
            row["n_with_thinking"] = has_thinking
            row["mean_thinking_chars"] = mean_thinking_len

        summary_rows.append(row)

    return summary_rows


def save_summary_csv(summary_rows: list[dict], output_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = output_dir / f"summary_{timestamp}.csv"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not summary_rows:
        return filepath

    fieldnames = list(summary_rows[0].keys())
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    logging.info("Summary saved to %s", filepath)
    return filepath


def print_summary_table(summary_rows: list[dict], mode: str) -> None:
    if not summary_rows:
        print("\nNo results to summarize.")
        return

    thinking = (mode == "thinking")
    label = "THINKING" if thinking else "BASELINE"

    print("\n" + "=" * 120)
    print(f"EVALUATION SUMMARY ({label})")
    print("=" * 120)

    header = (
        f"{'Model':<36} {'Dir':<6} {'N':>4} "
        f"{'chrF++':>7} {'Norm':>7} {'Δ':>5} "
        f"{'BLEU':>6} {'Conf':>6} "
        f"{'Ref':>4} {'Emp':>4} {'Err':>4}"
    )
    if thinking:
        header += f" {'Think':>6}"
    print(header)
    print("-" * 120)

    for row in summary_rows:
        delta = round(row["corpus_chrf_pp_norm"] - row["corpus_chrf_pp"], 1)
        conf_str = f"{row['mean_confidence']:.2f}" if row["mean_confidence"] is not None else "—"
        dir_label = "L→E" if row["direction"] == "lak_to_eng" else "E→L"

        line = (
            f"{row['model']:<36} {dir_label:<6} {row['n_translations']:>4} "
            f"{row['corpus_chrf_pp']:>7.1f} {row['corpus_chrf_pp_norm']:>7.1f} {delta:>+5.1f} "
            f"{row['corpus_bleu']:>6.1f} {conf_str:>6} "
            f"{row['n_refusals']:>4} {row['n_empties']:>4} {row['n_errors']:>4}"
        )
        if thinking:
            think_n = row.get("n_with_thinking", 0)
            think_str = f"{think_n}" if think_n else "—"
            line += f" {think_str:>6}"
        print(line)

    print("=" * 120)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate LLMs on Lakota-English translation.",
    )
    parser.add_argument("--thinking", action="store_true",
                        help="Enable extended thinking/reasoning (temp=1, higher token limits)")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY)
    parser.add_argument("--holdout-dir", type=Path, default=HOLDOUT_DIR)
    parser.add_argument("--results-dir", type=Path, default=None,
                        help="Output directory (default: results/baseline/ or results/thinking/)")
    parser.add_argument("--runs", type=int, default=1,
                        help="Number of evaluation runs (for multi-run variance)")
    parser.add_argument("--pilot", action="store_true",
                        help=f"Pilot mode: {DEFAULT_PILOT_PAIRS} pairs x {DEFAULT_PILOT_RUNS} runs")
    parser.add_argument("--pilot-pairs", type=int, default=DEFAULT_PILOT_PAIRS)
    parser.add_argument("--pilot-runs", type=int, default=DEFAULT_PILOT_RUNS)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--direction", choices=["lak_to_eng", "eng_to_lak"],
                        default=None, help="Run only one direction (default: both)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    mode = "thinking" if args.thinking else "baseline"
    defaults = THINKING_DEFAULTS if args.thinking else BASELINE_DEFAULTS
    temperature = defaults["temperature"]
    max_tokens = defaults["max_tokens"]
    timeout = defaults["timeout"]

    results_dir = args.results_dir or (PROJECT_ROOT / "results" / mode)

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format=LOG_FORMAT)

    if load_dotenv is not None:
        env_path = PROJECT_ROOT / ".env"
        if env_path.exists():
            load_dotenv(env_path)
            logging.info("Loaded environment from %s", env_path)

    if not args.verbose:
        litellm.suppress_debug_info = True
        logging.getLogger("LiteLLM").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)

    # Load data
    all_pairs = load_holdout_data(args.holdout_dir)

    # Pilot mode
    if args.pilot:
        random.seed(args.seed)
        pilot_pairs = random.sample(all_pairs, min(args.pilot_pairs, len(all_pairs)))
        logging.info("Pilot: sampled %d pairs (seed=%d)", len(pilot_pairs), args.seed)
        run_pilot(
            pilot_pairs, args.models, args.delay, args.pilot_runs,
            mode, temperature, max_tokens, timeout, results_dir,
        )
        return

    # Normal mode
    if args.sample is not None and args.sample < len(all_pairs):
        random.seed(args.seed)
        all_pairs = random.sample(all_pairs, args.sample)
        logging.info("Sampled %d pairs (seed=%d)", len(all_pairs), args.seed)

    n_dirs = 1 if args.direction else 2
    n_calls = len(all_pairs) * len(args.models) * n_dirs * args.runs
    logging.info("-" * 60)
    logging.info("Evaluation plan (%s):", mode.upper())
    logging.info("  Models:      %s", ", ".join(args.models))
    logging.info("  Pairs:       %d", len(all_pairs))
    logging.info("  Runs:        %d", args.runs)
    logging.info("  Total calls: %d", n_calls)
    logging.info("  Temperature: %d", temperature)
    logging.info("  Max tokens:  %d", max_tokens)
    if args.thinking:
        for m in args.models:
            logging.info("  Thinking [%s]: %s", m, describe_thinking(m))
    logging.info("  Output:      %s", results_dir)
    logging.info("  Delay:       %.1fs", args.delay)
    if not args.dry_run:
        est_per_call = args.delay + (5 if args.thinking else 3)
        est_minutes = n_calls * est_per_call / 60
        logging.info("  Est. time:   ~%.0f minutes", est_minutes)
    logging.info("-" * 60)

    # Run
    dirs = [args.direction] if args.direction else None
    results = run_evaluation(
        all_pairs, args.models, args.delay,
        mode, temperature, max_tokens, timeout,
        args.runs, args.dry_run, dirs,
    )

    # Save raw
    save_raw_results(results, results_dir)

    # Summary
    if not args.dry_run:
        summary = compute_summary(results)
        save_summary_csv(summary, results_dir)
        print_summary_table(summary, mode)

    # Save metadata
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    meta_path = results_dir / f"run_metadata_{timestamp}.json"
    results_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "run_id": str(uuid.uuid4()),
        "mode": mode,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dry_run": args.dry_run,
        "python_version": sys.version,
        "models_evaluated": args.models,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout": timeout,
        "num_runs": args.runs,
        "total_pairs": len(all_pairs),
        "sample_size": args.sample,
        "random_seed": args.seed,
        "system_prompt": SYSTEM_PROMPT,
        "response_schema": RESPONSE_SCHEMA,
    }
    if args.thinking:
        metadata["thinking_configs"] = {m: describe_thinking(m) for m in args.models}
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logging.info("Done.")


if __name__ == "__main__":
    main()

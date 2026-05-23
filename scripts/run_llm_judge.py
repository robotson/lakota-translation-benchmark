#!/usr/bin/env python3
"""
run_llm_judge.py -- LLM-as-Judge Semantic Equivalence Evaluation

Uses a cheap/fast LLM (Gemini Flash) to judge whether lak→eng model outputs
are semantically equivalent to reference translations. The judge only sees
two English strings — it needs no knowledge of Lakota.

This addresses a limitation of chrF++ (penalizes valid paraphrases) and
BERTScore (rewards fluent hallucinations) by providing a semantic equivalence
judgment that answers the actual question: did the model convey the same meaning?

Design:
- Reads lak→eng composite JSONL files (or BERTScore-enriched versions)
- Sends (reference, hypothesis) pairs to judge model
- Three-tier judgment: equivalent / partially_equivalent / not_equivalent
- Saves results iteratively (one JSONL line per judgment, appended)
- Fully resumable: skips pairs that already have judgments
- Configurable daily call limit (default 250) for rate-limit safety

Usage:
    python scripts/run_llm_judge.py                             # run all, default key
    python scripts/run_llm_judge.py --models gemini_baseline    # single model
    python scripts/run_llm_judge.py --max-calls 100             # limit API calls
    python scripts/run_llm_judge.py --dry-run                   # preview only
    python scripts/run_llm_judge.py --api-key AIza...           # use alternate Gemini key
    python scripts/run_llm_judge.py --judge-model gemini/gemini-2.5-flash  # override judge

Key rotation for rate limits:
    python scripts/run_llm_judge.py --max-calls 500             # key 1 (from .env)
    python scripts/run_llm_judge.py --max-calls 500 --api-key AIza...  # key 2
    # Script is resumable — each run picks up where the last left off.

Requires:
    - GEMINI_API_KEY in .env (or pass --api-key)
    - pip install litellm python-dotenv
"""

import argparse
import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

try:
    import litellm
except ImportError:
    sys.exit("ERROR: litellm not installed. Run: pip install litellm")

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMPOSITE_DIR = PROJECT_ROOT / "results" / "composite"

DEFAULT_JUDGE_MODEL = "gemini/gemini-3-flash-preview"
DEFAULT_DELAY = 2.0       # seconds between calls
DEFAULT_TIMEOUT = 60
DEFAULT_MAX_RETRIES = 3
DEFAULT_MAX_CALLS = 250   # conservative daily limit
DEFAULT_TEMPERATURE = 1   # Gemini 3 models require temp=1 for reliable output

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"

SYSTEM_PROMPT = """\
You are an expert linguistic judge. Your task is to determine whether two English \
sentences convey the same meaning.

You will receive a reference sentence and a candidate sentence. Judge whether the \
candidate conveys the same meaning as the reference. Minor differences in wording, \
formality, or sentence structure that preserve the core meaning should be judged as \
equivalent. Significant omissions, additions, or changes in meaning should be judged \
as not equivalent.

Respond as JSON with two fields:
- "judgment": one of "equivalent", "partially_equivalent", or "not_equivalent"
- "reason": a brief (1-2 sentence) explanation of your judgment

Definitions:
- equivalent: The candidate conveys the same core meaning as the reference, even if \
worded differently. Example: "Do you have kids?" vs "Do you have children?" → equivalent
- partially_equivalent: The candidate captures some of the meaning but is missing \
important elements, adds significant meaning not in the reference, or shifts the \
emphasis substantially. Example: "I should go" vs "I have to go there" → partially \
(missing "there", softened obligation)
- not_equivalent: The candidate conveys a different meaning from the reference. \
Example: "Do you use sugar?" vs "Is that coffee?" → not_equivalent"""

USER_PROMPT_TEMPLATE = """\
Reference: {reference}
Candidate: {candidate}"""

RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "semantic_judgment",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "judgment": {
                    "type": "string",
                    "enum": ["equivalent", "partially_equivalent", "not_equivalent"],
                },
                "reason": {"type": "string"},
            },
            "required": ["judgment", "reason"],
            "additionalProperties": False,
        },
    },
}


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

def call_judge(
    judge_model: str,
    reference: str,
    candidate: str,
    temperature: float = DEFAULT_TEMPERATURE,
    timeout: int = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> dict:
    """Call judge LLM. Returns dict with judgment, reason, metadata."""
    start_time = time.monotonic()
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                wait = min(2 ** attempt, 30)
                logging.info("    Retry %d/%d after %ds...", attempt, max_retries, wait)
                time.sleep(wait)

            user_prompt = USER_PROMPT_TEMPLATE.format(
                reference=reference, candidate=candidate
            )

            response = litellm.completion(
                model=judge_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=256,
                timeout=timeout,
                response_format=RESPONSE_SCHEMA,
            )

            latency_ms = int((time.monotonic() - start_time) * 1000)

            # Handle thinking models: content may be None, check for parts
            raw_content = response.choices[0].message.content
            thinking_text = None

            # If content is None, try to extract from tool_calls or other fields
            if raw_content is None:
                # Some thinking models put content in a different place
                # Try the _hidden_params or model_extra approach
                msg = response.choices[0].message
                if hasattr(msg, 'model_extra') and msg.model_extra:
                    parts = msg.model_extra.get('parts', [])
                    for part in parts:
                        if hasattr(part, 'text') and part.text:
                            raw_content = part.text
                        elif isinstance(part, dict):
                            if part.get('text'):
                                raw_content = part['text']
                            elif part.get('thought'):
                                thinking_text = part['thought']

            # Still None? Try raw response
            if raw_content is None:
                raw_dict = response.to_dict() if hasattr(response, 'to_dict') else {}
                # Walk the structure looking for text content
                try:
                    candidates = raw_dict.get('candidates', [{}])
                    if candidates:
                        parts = candidates[0].get('content', {}).get('parts', [])
                        for part in parts:
                            if 'text' in part:
                                raw_content = part['text']
                            elif 'thought' in part:
                                thinking_text = part['thought']
                except (KeyError, IndexError, TypeError):
                    pass

            raw_response = raw_content or ""

            usage = response.usage if response.usage else None
            input_tokens = usage.prompt_tokens if usage else None
            output_tokens = usage.completion_tokens if usage else None

            # Parse JSON
            parsed = None
            parse_error = None
            if raw_response:
                try:
                    parsed = json.loads(raw_response)
                except json.JSONDecodeError as e:
                    parse_error = str(e)

            judgment = None
            reason = None
            if parsed:
                judgment = parsed.get("judgment")
                reason = parsed.get("reason")
                # Validate judgment value
                if judgment not in ("equivalent", "partially_equivalent", "not_equivalent"):
                    parse_error = f"Invalid judgment value: {judgment}"
                    judgment = None

            return {
                "judgment": judgment,
                "reason": reason,
                "raw_response": raw_response,
                "thinking_text": thinking_text,
                "parse_error": parse_error,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "latency_ms": latency_ms,
                "error": None,
                "attempts": attempt + 1,
            }

        except Exception as e:
            last_error = e
            err_str = str(e).lower()
            if "timeout" in err_str or "rate" in err_str or "429" in err_str:
                continue  # retry on timeout or rate limit
            else:
                break  # don't retry on other errors

    latency_ms = int((time.monotonic() - start_time) * 1000)
    return {
        "judgment": None,
        "reason": None,
        "raw_response": "",
        "thinking_text": None,
        "parse_error": None,
        "input_tokens": None,
        "output_tokens": None,
        "latency_ms": latency_ms,
        "error": str(last_error),
        "attempts": max_retries + 1,
    }


# ---------------------------------------------------------------------------
# Data loading and resumption
# ---------------------------------------------------------------------------

def get_hyp(record: dict) -> str | None:
    """Get model output, falling back to raw_response if translation is None."""
    if record.get("translation") and record["translation"].strip():
        return record["translation"]
    if record.get("raw_response") and record["raw_response"].strip():
        return record["raw_response"]
    return None


def load_composite_pairs(model_dir: str) -> list[dict]:
    """Load lak_to_eng composite JSONL for a model."""
    # Prefer BERTScore-enriched version (has all fields + BERTScore)
    bs_path = COMPOSITE_DIR / model_dir / "lak_to_eng_bertscore.jsonl"
    plain_path = COMPOSITE_DIR / model_dir / "lak_to_eng.jsonl"

    fpath = bs_path if bs_path.exists() else plain_path
    if not fpath.exists():
        logging.warning("No lak_to_eng file found for %s", model_dir)
        return []

    records = []
    with open(fpath) as f:
        for line in f:
            r = json.loads(line)
            if get_hyp(r) and r.get("response_type") == "translation":
                records.append(r)

    logging.info("Loaded %d translatable pairs from %s", len(records), fpath.name)
    return records


def load_existing_judgments(outpath: Path) -> set[tuple[str, int]]:
    """Load already-judged (model_dir, pair_index) tuples from output file."""
    done = set()
    if outpath.exists():
        with open(outpath) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add((r["evaluated_model"], r["pair_index"]))
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LLM-as-Judge semantic equivalence evaluation")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL,
                        help=f"Judge model (default: {DEFAULT_JUDGE_MODEL})")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Model dirs to evaluate (default: all in composite/)")
    parser.add_argument("--max-calls", type=int, default=DEFAULT_MAX_CALLS,
                        help=f"Max API calls per run (default: {DEFAULT_MAX_CALLS})")
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY,
                        help=f"Delay between API calls in seconds (default: {DEFAULT_DELAY})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview what would be judged without calling API")
    parser.add_argument("--api-key", default=None,
                        help="Override GEMINI_API_KEY (for key rotation)")
    parser.add_argument("--output", default=None,
                        help="Output JSONL path (default: results/composite/llm_judge_results.jsonl)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    if load_dotenv:
        load_dotenv(PROJECT_ROOT / ".env")

    # Override API key if provided
    if args.api_key:
        os.environ["GEMINI_API_KEY"] = args.api_key
        logging.info("Using provided API key (override)")

    # Discover model directories
    if args.models:
        model_dirs = args.models
    else:
        model_dirs = sorted([
            d.name for d in COMPOSITE_DIR.iterdir()
            if d.is_dir() and (d / "lak_to_eng.jsonl").exists()
        ])

    logging.info("Judge model: %s", args.judge_model)
    logging.info("Model dirs to evaluate: %s", model_dirs)
    logging.info("Max calls this run: %d", args.max_calls)

    # Output file
    outpath = Path(args.output) if args.output else COMPOSITE_DIR / "llm_judge_results.jsonl"
    logging.info("Output: %s", outpath)

    # Load existing judgments for resumption
    done = load_existing_judgments(outpath)
    logging.info("Already judged: %d pairs", len(done))

    # Collect all work to do
    work = []
    for model_dir in model_dirs:
        pairs = load_composite_pairs(model_dir)
        for rec in pairs:
            pair_key = (model_dir, rec["pair_index"])
            if pair_key not in done:
                work.append((model_dir, rec))

    logging.info("Pairs remaining: %d", len(work))
    logging.info("Will process up to %d this run", min(len(work), args.max_calls))

    if args.dry_run:
        for model_dir, rec in work[:10]:
            hyp = get_hyp(rec)
            print(f"  {model_dir} pair {rec['pair_index']}: "
                  f"ref={rec['reference_text'][:50]}... hyp={hyp[:50]}...")
        if len(work) > 10:
            print(f"  ... and {len(work) - 10} more")
        return

    # Run judgments
    call_count = 0
    error_count = 0
    results_this_run = {"equivalent": 0, "partially_equivalent": 0, "not_equivalent": 0, "error": 0}

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    for model_dir, rec in work:
        if call_count >= args.max_calls:
            logging.info("Reached max calls (%d). Stopping. Resume later to continue.",
                         args.max_calls)
            break

        pair_idx = rec["pair_index"]
        reference = rec["reference_text"]
        candidate = get_hyp(rec)

        logging.info("[%d/%d] %s pair %d", call_count + 1, min(len(work), args.max_calls),
                     model_dir, pair_idx)

        result = call_judge(
            judge_model=args.judge_model,
            reference=reference,
            candidate=candidate,
            timeout=DEFAULT_TIMEOUT,
        )

        # Build output record
        out_record = {
            "run_id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "judge_model": args.judge_model,
            "evaluated_model": model_dir,
            "pair_index": pair_idx,
            "direction": "lak_to_eng",
            "source_text": rec.get("source_text", ""),
            "reference_text": reference,
            "candidate_text": candidate,
            "chrf_pp": rec.get("chrf_pp"),
            "bertscore_f1": rec.get("bertscore_f1"),
            "judgment": result["judgment"],
            "reason": result["reason"],
            "judge_raw_response": result["raw_response"],
            "judge_thinking": result["thinking_text"],
            "judge_parse_error": result["parse_error"],
            "judge_error": result["error"],
            "judge_latency_ms": result["latency_ms"],
            "judge_attempts": result["attempts"],
            "judge_input_tokens": result["input_tokens"],
            "judge_output_tokens": result["output_tokens"],
        }

        # Append to output file (iterative save)
        with open(outpath, "a") as f:
            f.write(json.dumps(out_record) + "\n")

        call_count += 1

        if result["judgment"]:
            results_this_run[result["judgment"]] += 1
            logging.info("    → %s (chrF++=%s, BERTScore=%s): %s",
                         result["judgment"],
                         rec.get("chrf_pp", "?"),
                         f"{rec.get('bertscore_f1', 0):.3f}" if rec.get("bertscore_f1") else "?",
                         result["reason"][:80] if result["reason"] else "")
        else:
            results_this_run["error"] += 1
            error_count += 1
            logging.warning("    → ERROR: %s", result["error"] or result["parse_error"])

        if call_count < min(len(work), args.max_calls):
            time.sleep(args.delay)

    # Summary
    total_done = len(done) + call_count - error_count
    total_work = len(done) + len(work)
    logging.info("")
    logging.info("=== Run Summary ===")
    logging.info("Calls made: %d", call_count)
    logging.info("This run: %s", dict(results_this_run))
    logging.info("Progress: %d / %d pairs judged (%.1f%%)",
                 total_done, total_work, 100 * total_done / total_work if total_work else 0)
    if total_done < total_work:
        remaining = total_work - total_done
        logging.info("Remaining: %d pairs. Run again to continue.", remaining)


if __name__ == "__main__":
    main()

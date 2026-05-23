#!/usr/bin/env python3
"""run_llm_judge_llama.py -- independent second judge (Llama-3.3-70B via Together).

A second, independent semantic-equivalence judge from a different model family
and vendor than the primary Gemini judge (run_llm_judge.py), used to measure
inter-judge agreement and rule out single-judge / family-level bias.

Contract-identical by import: same SYSTEM_PROMPT, USER_PROMPT_TEMPLATE,
RESPONSE_SCHEMA, load_composite_pairs, get_hyp, load_existing_judgments, and
output-record schema as the Gemini judge -- so agreement is apples-to-apples.
Writes a separate file (llm_judge_results_llama.jsonl) with its own resume
tracking, so it can never collide with or disturb the Gemini judgments.

temperature=0 here (deterministic, appropriate for a judge); the Gemini judge
uses temperature=1 only because Gemini 3 requires it.

Usage:
    python scripts/run_llm_judge_llama.py --dry-run
    python scripts/run_llm_judge_llama.py
    python scripts/run_llm_judge_llama.py --models gemini_thinking deepseek_thinking

Requires:
    - TOGETHER_API_KEY in .env
    - Composite results in results/composite/<model>/lak_to_eng.jsonl
    - pip install litellm python-dotenv
"""

import argparse
import json
import logging
import os
import sys
import time
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
if load_dotenv:
    load_dotenv(ROOT / ".env")

from run_llm_judge import (
    SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, RESPONSE_SCHEMA,
    load_composite_pairs, get_hyp, load_existing_judgments, COMPOSITE_DIR,
)

TOGETHER_BASE = "https://api.together.xyz/v1"
TOGETHER_KEY = os.getenv("TOGETHER_API_KEY")
JUDGE_MODEL = "openai/meta-llama/Llama-3.3-70B-Instruct-Turbo"
OUT_PATH = COMPOSITE_DIR / "llm_judge_results_llama.jsonl"
MAX_RETRIES = 3
TIMEOUT = 90
VALID = ("equivalent", "partially_equivalent", "not_equivalent")


def call_judge_llama(reference: str, candidate: str) -> dict:
    start = time.monotonic()
    last_err = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            if attempt:
                time.sleep(min(2 ** attempt, 30))
            resp = litellm.completion(
                model=JUDGE_MODEL, api_base=TOGETHER_BASE, api_key=TOGETHER_KEY,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PROMPT_TEMPLATE.format(
                        reference=reference, candidate=candidate)},
                ],
                temperature=0, max_tokens=256, timeout=TIMEOUT,
                response_format=RESPONSE_SCHEMA)
            raw = resp.choices[0].message.content or ""
            u = resp.usage if resp.usage else None
            parsed, perr = None, None
            if raw:
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError as e:
                    perr = str(e)
            judgment = reason = None
            if parsed:
                judgment = parsed.get("judgment")
                reason = parsed.get("reason")
                if judgment not in VALID:
                    perr = f"Invalid judgment value: {judgment}"
                    judgment = None
            return {
                "judgment": judgment, "reason": reason, "raw_response": raw,
                "parse_error": perr,
                "input_tokens": u.prompt_tokens if u else None,
                "output_tokens": u.completion_tokens if u else None,
                "latency_ms": int((time.monotonic() - start) * 1000),
                "error": None, "attempts": attempt + 1,
            }
        except Exception as e:
            last_err = e
            es = str(e).lower()
            if "timeout" in es or "rate" in es or "429" in es:
                continue
            break
    return {
        "judgment": None, "reason": None, "raw_response": "", "parse_error": None,
        "input_tokens": None, "output_tokens": None,
        "latency_ms": int((time.monotonic() - start) * 1000),
        "error": str(last_err), "attempts": MAX_RETRIES + 1,
    }


def main():
    ap = argparse.ArgumentParser(description="Independent Llama-3.3-70B judge.")
    ap.add_argument("--models", nargs="*", default=None,
                    help="composite dir names to judge (default: all)")
    ap.add_argument("--max-calls", type=int, default=5000)
    ap.add_argument("--delay", type=float, default=0.2)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    if not TOGETHER_KEY and not a.dry_run:
        sys.exit("ERROR: TOGETHER_API_KEY missing in .env")

    model_dirs = a.models or sorted(
        d.name for d in COMPOSITE_DIR.iterdir()
        if d.is_dir() and (d / "lak_to_eng.jsonl").exists())
    done = load_existing_judgments(OUT_PATH)
    work = []
    for md in model_dirs:
        for rec in load_composite_pairs(md):
            if (md, rec["pair_index"]) not in done:
                work.append((md, rec))
    logging.info("judge=%s | dirs=%d | already=%d | remaining=%d | cap=%d",
                 JUDGE_MODEL, len(model_dirs), len(done), len(work), a.max_calls)
    if a.dry_run:
        for md, rec in work[:8]:
            print(f"  {md} p{rec['pair_index']} ref={rec['reference_text'][:45]!r} "
                  f"hyp={(get_hyp(rec) or '')[:45]!r}")
        print(f"  ...+{max(0, len(work) - 8)} more")
        return

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    n = 0
    tally = {"equivalent": 0, "partially_equivalent": 0, "not_equivalent": 0, "error": 0}
    for md, rec in work:
        if n >= a.max_calls:
            logging.info("hit cap %d; resume to continue", a.max_calls)
            break
        ref, cand = rec["reference_text"], get_hyp(rec)
        r = call_judge_llama(ref, cand)
        out = {
            "run_id": run_id, "timestamp": datetime.now(timezone.utc).isoformat(),
            "judge_model": JUDGE_MODEL, "evaluated_model": md,
            "pair_index": rec["pair_index"], "direction": "lak_to_eng",
            "source_text": rec.get("source_text", ""),
            "reference_text": ref, "candidate_text": cand,
            "chrf_pp": rec.get("chrf_pp"), "bertscore_f1": rec.get("bertscore_f1"),
            "judgment": r["judgment"], "reason": r["reason"],
            "judge_raw_response": r["raw_response"], "judge_parse_error": r["parse_error"],
            "judge_error": r["error"], "judge_latency_ms": r["latency_ms"],
            "judge_attempts": r["attempts"],
            "judge_input_tokens": r["input_tokens"], "judge_output_tokens": r["output_tokens"],
        }
        with open(OUT_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
        n += 1
        tally[r["judgment"] or "error"] = tally.get(r["judgment"] or "error", 0) + 1
        if n % 25 == 0:
            logging.info("%d/%d done | tally=%s", n, min(len(work), a.max_calls), tally)
        time.sleep(a.delay)
    logging.info("DONE n=%d tally=%s -> %s", n, tally, OUT_PATH)


if __name__ == "__main__":
    main()

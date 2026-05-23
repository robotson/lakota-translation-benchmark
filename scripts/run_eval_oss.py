#!/usr/bin/env python3
"""run_eval_oss.py -- Open-weight model eval runner (DeepSeek, GLM, Qwen).

Companion to run_eval.py for the three open-weight models in the study,
served via Together AI. To keep cross-model comparison apples-to-apples,
this imports the EXACT structured-output schema, prompts, response
classification, and metric functions from run_eval -- the same scoring
pipeline used for the proprietary models. Differences are confined to
transport and per-model reasoning toggles.

Transport: routed as a generic OpenAI-compatible endpoint to Together
(api_base + api_key), which preserves the strict json_schema contract.

Two conditions per model via --mode {thinking,nonthinking}:
  - thinking: provider's reasoning left on (its default for these models).
  - nonthinking: reasoning disabled via each model's verified disabler.
    This is a clean within-model ablation -- only reasoning is toggled;
    the output-format regime is held identical to that model's thinking
    condition, so baseline-vs-thinking matches the proprietary design.

Verified disablers (others silently fail -- see notes inline):
  - DeepSeek: reasoning={"enabled": False}
  - Qwen / GLM: extra_body chat_template_kwargs={"enable_thinking": False}
Qwen surfaces no reasoning field, so thinking-off is evidenced by the
token/latency collapse recorded per record, not by reasoning length.

GLM note: Together/vLLM guided decoding suppresses GLM's reasoning whenever
any response_format is set, so GLM uses prompt-elicited JSON (the system
prompt already specifies the 3-field format) in BOTH conditions -- parsed
locally. DeepSeek and Qwen use provider-enforced json_schema in both.

Resumable: appends per (model, direction); rerunning skips done pair_index.

Usage:
    python scripts/run_eval_oss.py --dry-run
    python scripts/run_eval_oss.py --mode thinking
    python scripts/run_eval_oss.py --mode nonthinking
    python scripts/run_eval_oss.py --models DeepSeek-V4-Pro --limit 5

Requires:
    - TOGETHER_API_KEY in .env
    - Holdout pairs in data/holdout/ (see data/example_pairs.json)
    - pip install litellm sacrebleu python-dotenv
"""

import argparse
import json
import os
import re
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
if load_dotenv:
    load_dotenv(PROJECT_ROOT / ".env")

# Shared contract with the proprietary runner -- identical scoring pipeline.
from run_eval import (
    RESPONSE_SCHEMA,
    SYSTEM_PROMPT,
    USER_PROMPT_LAK_TO_ENG,
    USER_PROMPT_ENG_TO_LAK,
    classify_response,
    compute_chrf_pp,
    compute_bleu,
    load_holdout_data,
    HOLDOUT_DIR,
)

TOGETHER_BASE = "https://api.together.xyz/v1"
TOGETHER_KEY = os.getenv("TOGETHER_API_KEY")
OUT_DIR = PROJECT_ROOT / "results" / "oss" / "raw"

# Per-model transport config. `structured`=True uses provider-enforced
# json_schema; GLM is False (see module docstring -- json_schema suppresses
# GLM reasoning, so it uses prompt-elicited JSON parsed locally).
MODELS = {
    "GLM-5.1": {
        "model": "openai/zai-org/GLM-5.1",
        "max_tokens": 32768, "stream": False, "timeout": 360,
        "structured": False,
    },
    "Qwen3.6-Plus": {
        "model": "openai/Qwen/Qwen3.6-Plus",
        "max_tokens": 2048, "stream": True, "timeout": 300,
        "structured": True,  # stream-only + heavy always-on reasoning
    },
    "DeepSeek-V4-Pro": {
        "model": "openai/deepseek-ai/DeepSeek-V4-Pro",
        "max_tokens": 16384, "stream": False, "timeout": 240,
        "structured": True,  # heavy reasoning; <16k can truncate to empty
    },
}

# Non-thinking ("baseline") overrides. Each disabler is the one verified to
# actually toggle reasoning off for that model under this exact contract;
# format regime is held identical to the model's thinking condition.
# max_tokens=4096: non-thinking answers are short, but a few hard pairs can
# trigger a degenerate repetition loop -- the larger cap ensures a parse
# failure reflects a genuine model failure, not a truncated output.
NONTHINKING = {
    "GLM-5.1": {
        "max_tokens": 4096, "stream": False, "timeout": 120, "structured": False,
        "extra_kw": {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
    },
    "Qwen3.6-Plus": {
        "max_tokens": 4096, "stream": True, "timeout": 180, "structured": True,
        "extra_kw": {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
    },
    "DeepSeek-V4-Pro": {
        "max_tokens": 4096, "stream": False, "timeout": 120, "structured": True,
        "extra_kw": {"reasoning": {"enabled": False}},
    },
}

MAX_RETRIES = 3
_FENCE = re.compile(r"```(?:json)?\s*|\s*```")


def extract_json(raw):
    """Parse JSON from possibly fenced / reasoning-prefixed content."""
    if not raw:
        return None, "empty"
    try:
        return json.loads(raw), None
    except json.JSONDecodeError:
        pass
    s = _FENCE.sub("", raw)
    try:
        return json.loads(s.strip()), None
    except json.JSONDecodeError:
        pass
    i = s.find("{")
    if i != -1:
        depth = 0
        for j in range(i, len(s)):
            if s[j] == "{":
                depth += 1
            elif s[j] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(s[i:j + 1]), None
                    except json.JSONDecodeError as e:
                        return None, str(e)
    return None, "no_json_object_found"


def call_oss(cfg, system_prompt, user_prompt):
    """Call one OSS model via Together, handling stream-only models and a
    hard streaming deadline (some models trickle indefinitely on hard pairs)."""
    start = time.monotonic()
    last_err = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            if attempt:
                time.sleep(min(2 ** attempt, 30))
            kw = dict(
                model=cfg["model"], api_base=TOGETHER_BASE, api_key=TOGETHER_KEY,
                messages=[{"role": "system", "content": system_prompt},
                          {"role": "user", "content": user_prompt}],
                temperature=0, max_tokens=cfg["max_tokens"], timeout=cfg["timeout"],
            )
            if cfg.get("structured", True):
                kw["response_format"] = RESPONSE_SCHEMA
            kw.update(cfg.get("extra_kw") or {})  # non-thinking disabler

            stream = cfg["stream"]
            reasoning = None
            if not stream:
                try:
                    r = litellm.completion(**kw)
                except Exception as e:
                    if "only supports streaming" not in str(e).lower():
                        raise
                    stream = True  # fall through to streaming path

            if stream:
                kw["stream"] = True
                kw["stream_options"] = {"include_usage": True}
                buf, rbuf, usage, finish = [], [], None, None
                deadline = time.monotonic() + cfg["timeout"]
                for ch in litellm.completion(**kw):
                    if time.monotonic() > deadline:
                        raise TimeoutError(f"hard stream deadline {cfg['timeout']}s exceeded")
                    if ch.choices:
                        d = ch.choices[0].delta
                        if getattr(d, "content", None):
                            buf.append(d.content)
                        if getattr(d, "reasoning", None):
                            rbuf.append(d.reasoning)
                        if ch.choices[0].finish_reason:
                            finish = ch.choices[0].finish_reason
                    if getattr(ch, "usage", None):
                        usage = ch.usage
                raw = "".join(buf)
                reasoning = "".join(rbuf) or None
            else:
                msg = r.choices[0].message
                raw = msg.content or ""
                reasoning = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None)
                finish = r.choices[0].finish_reason
                usage = r.usage if r.usage else None

            in_tok = usage.prompt_tokens if usage else None
            out_tok = usage.completion_tokens if usage else None
            parsed, perr = extract_json(raw)
            return {
                "raw_response": raw, "parsed": parsed, "parse_error": perr,
                "finish_reason": finish, "input_tokens": in_tok, "output_tokens": out_tok,
                "reasoning_chars": len(reasoning) if reasoning else 0,
                "latency_ms": int((time.monotonic() - start) * 1000),
                "error": None, "attempts": attempt + 1,
            }
        except Exception as e:
            last_err = e
            if "timeout" in str(e).lower() or "rate" in str(e).lower():
                continue
            break
    return {
        "raw_response": "", "parsed": None, "parse_error": None,
        "finish_reason": None, "input_tokens": None, "output_tokens": None,
        "reasoning_chars": 0, "latency_ms": int((time.monotonic() - start) * 1000),
        "error": str(last_err), "attempts": MAX_RETRIES + 1,
    }


def done_set(path):
    """Pair indices already recorded in a JSONL file (for resumability)."""
    s = set()
    if path.exists():
        for ln in path.read_text(encoding="utf-8").splitlines():
            try:
                s.add(json.loads(ln)["pair_index"])
            except Exception:
                pass
    return s


def main():
    ap = argparse.ArgumentParser(description="OSS-model translation eval (Together).")
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--mode", choices=("thinking", "nonthinking"), default="thinking",
                    help="thinking = reasoning on (provider default); "
                         "nonthinking = reasoning disabled (clean ablation, separate dir)")
    ap.add_argument("--limit", type=int, help="evaluate only the first N pairs")
    ap.add_argument("--delay", type=float, default=1.0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if not TOGETHER_KEY and not a.dry_run:
        sys.exit("ERROR: TOGETHER_API_KEY missing in .env")

    pairs = load_holdout_data(HOLDOUT_DIR)
    if a.limit:
        pairs = pairs[: a.limit]

    for name in a.models:
        cfg = dict(MODELS[name])
        if a.mode == "nonthinking":
            cfg.update(NONTHINKING[name])
        safe = name.replace("/", "_")
        mdir = OUT_DIR / safe / a.mode
        mdir.mkdir(parents=True, exist_ok=True)
        for direction in ("lak_to_eng", "eng_to_lak"):
            fp = mdir / f"{direction}.jsonl"
            done = done_set(fp)
            todo = [p for p in pairs if p["pair_index"] not in done]
            print(f"[{name}/{a.mode}] {direction}: {len(done)} done, {len(todo)} to do -> {fp}")
            if a.dry_run:
                continue
            with fp.open("a", encoding="utf-8") as out:
                for n, p in enumerate(todo):
                    if direction == "lak_to_eng":
                        src, ref = p["lakota"], p["english"]
                        up = USER_PROMPT_LAK_TO_ENG.format(text=src)
                    else:
                        src, ref = p["english"], p["lakota"]
                        up = USER_PROMPT_ENG_TO_LAK.format(text=src)
                    res = call_oss(cfg, SYSTEM_PROMPT, up)
                    parsed = res["parsed"]
                    translation = (parsed or {}).get("translation", "") if parsed else ""
                    rtype = classify_response(parsed, res["raw_response"], res["error"])
                    rec = {
                        "id": str(uuid.uuid4()),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "model": name, "mode": a.mode,
                        "structured": cfg.get("structured", True),
                        "off_params": cfg.get("extra_kw"),
                        "direction": direction,
                        "source_text": src, "reference_text": ref,
                        "system_prompt": SYSTEM_PROMPT, "user_prompt": up,
                        "temperature": 0, "max_tokens": cfg["max_tokens"],
                        "source_file": p["source_file"], "source_name": p["source_name"],
                        "register": p["register"], "pair_index": p["pair_index"],
                        "context": p["context"],
                        "raw_response": res["raw_response"], "parsed": parsed,
                        "parse_error": res["parse_error"], "finish_reason": res["finish_reason"],
                        "translation": translation,
                        "confidence": (parsed or {}).get("confidence") if parsed else None,
                        "refusal_reason": (parsed or {}).get("refusal_reason") if parsed else None,
                        "response_type": rtype,
                        "chrf_pp": compute_chrf_pp(translation, ref) if translation else None,
                        "bleu": compute_bleu(translation, ref) if translation else None,
                        "input_tokens": res["input_tokens"], "output_tokens": res["output_tokens"],
                        "reasoning_chars": res["reasoning_chars"],
                        "latency_ms": res["latency_ms"],
                        "error": res["error"], "attempts": res["attempts"],
                    }
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out.flush()
                    if (n + 1) % 10 == 0:
                        print(f"  [{name}/{direction}] {n + 1}/{len(todo)} "
                              f"(last chrF++={rec['chrf_pp']}, {rec['response_type']})")
                    time.sleep(a.delay)
    print("Done.")


if __name__ == "__main__":
    main()

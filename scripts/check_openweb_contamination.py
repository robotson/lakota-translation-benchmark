#!/usr/bin/env python3
"""check_openweb_contamination.py -- open-web overlap audit for the eval set.

Estimates how recoverable the evaluation pairs are from public web content, to
bound data-contamination risk. Protocol:
  1. Query each Lakota sentence VERBATIM, quoted, ALONE (no English) via Serper.
  2. Tier 1: does the Lakota phrase appear verbatim in any organic result
     (title/snippet)? (matched on a diacritic-normalized form)
  3. Tier 2: for Tier-1 hits only, fetch the result page(s) and test whether
     the English REFERENCE translation co-occurs on the same source. Lakota +
     its English gloss on one page = aligned-pair (both-sided) overlap, the
     kind most associated with score inflation in controlled studies.
  4. Aggregate: % Lakota present anywhere; of those, % English co-located.

Resumable: appends one JSON line per pair; rerun skips recorded pairs.
Caveats (state in any writeup): Google index != training corpus; Google may
normalize diacritics; snippet/page text is a proxy; PDFs are flagged, not
deep-parsed.

Usage:
    python scripts/check_openweb_contamination.py --dry-run
    python scripts/check_openweb_contamination.py --limit 20
    python scripts/check_openweb_contamination.py

Requires:
    - SERPER_API_KEY in .env (https://serper.dev)
    - Holdout pairs in data/holdout/ (see data/example_pairs.json)
    - pip install requests python-dotenv
"""

import argparse
import ipaddress
import json
import os
import re
import socket
import sys
import time
import unicodedata
from pathlib import Path
from urllib.parse import urlparse

try:
    import requests
except ImportError:
    sys.exit("ERROR: requests not installed. Run: pip install requests")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

from run_eval import load_holdout_data, HOLDOUT_DIR

SERPER_URL = "https://google.serper.dev/search"
OUT_DIR = PROJECT_ROOT / "results" / "contamination"
OUT_JSONL = OUT_DIR / "serper_openweb.jsonl"
UA = "Mozilla/5.0 (research contamination audit; contact via paper)"
# Tier-2 fetch hardening: search-result URLs are attacker-influenceable (SEO),
# so cap blast radius -- http(s) only, no SSRF to internal hosts, bounded
# redirects and response size.
MAX_FETCH_BYTES = 3_000_000
MAX_REDIRECTS = 3
FETCH_TIMEOUT = 15


def _url_is_safe(url: str) -> bool:
    """Allow only http(s) to public hosts (SSRF guard). Reject if ANY
    resolved address is private/loopback/link-local/reserved."""
    try:
        p = urlparse(url)
        if p.scheme not in ("http", "https") or not p.hostname:
            return False
        infos = socket.getaddrinfo(p.hostname, p.port or 443, proto=socket.IPPROTO_TCP)
        for *_, sockaddr in infos:
            ip = ipaddress.ip_address(sockaddr[0])
            if (ip.is_private or ip.is_loopback or ip.is_link_local
                    or ip.is_reserved or ip.is_multicast or ip.is_unspecified):
                return False
        return bool(infos)
    except (ValueError, socket.gaierror, UnicodeError):
        return False


def norm(s: str) -> str:
    s = unicodedata.normalize("NFD", s.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", s)).strip()


def primary_lakota(lk: str) -> str:
    """Drop curated alternants/parentheticals: 'A / B', 'A (B)' -> 'A'."""
    lk = re.split(r"\s*[/(]", lk)[0]
    return lk.strip().rstrip(".").strip()


def serper(query: str, key: str, retries: int = 3):
    body = {"q": query, "gl": "us", "hl": "en", "num": 10}
    headers = {"X-API-KEY": key, "Content-Type": "application/json"}
    last = ""
    for attempt in range(retries):
        try:
            r = requests.post(SERPER_URL, json=body, headers=headers, timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503):
                time.sleep(2 * (attempt + 1))
                continue
            return {"_error": f"HTTP {r.status_code}: {r.text[:200]}"}
        except requests.RequestException as e:
            last = str(e)
            time.sleep(2 * (attempt + 1))
    return {"_error": f"request failed: {last}"}


def fetch_page_text(url: str):
    if not _url_is_safe(url):
        return None  # blocked: non-http(s) scheme or internal/SSRF target
    try:
        sess = requests.Session()
        sess.max_redirects = MAX_REDIRECTS
        r = sess.get(url, headers={"User-Agent": UA}, timeout=FETCH_TIMEOUT, stream=True)
        ct = r.headers.get("content-type", "")
        if "pdf" in ct.lower() or url.lower().endswith(".pdf"):
            r.close()
            return "__PDF__"  # flagged; deep PDF extraction left for manual follow-up
        if r.status_code != 200:
            r.close()
            return None
        chunks, total = [], 0
        for chunk in r.iter_content(chunk_size=65536, decode_unicode=False):
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_FETCH_BYTES:
                break  # bound memory / decompression-bomb exposure
            chunks.append(chunk)
        r.close()
        body = b"".join(chunks).decode("utf-8", errors="replace")
        text = re.sub(r"<script.*?</script>|<style.*?</style>", " ", body, flags=re.S | re.I)
        return re.sub(r"<[^>]+>", " ", text)
    except requests.RequestException:
        return None


def main():
    ap = argparse.ArgumentParser(description="Open-web overlap audit (Serper).")
    ap.add_argument("--limit", type=int, default=None, help="cap #pairs (default all)")
    ap.add_argument("--delay", type=float, default=0.3, help="sec between queries")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    key = os.getenv("SERPER_API_KEY")
    if not key and not args.dry_run:
        sys.exit("ERROR: SERPER_API_KEY not set in .env (https://serper.dev)")

    pairs = load_holdout_data(HOLDOUT_DIR)
    if args.limit:
        pairs = pairs[: args.limit]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    done = set()
    if OUT_JSONL.exists():
        for ln in OUT_JSONL.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(ln)["idx"])
            except Exception:
                pass
    print(f"{len(pairs)} pairs | {len(done)} already done | output -> {OUT_JSONL}")

    if args.dry_run:
        for p in pairs[:5]:
            print(f"  [{p['pair_index']}] query: \"{primary_lakota(p['lakota'])}\"")
        return

    fout = OUT_JSONL.open("a", encoding="utf-8")
    n_lak = n_both = 0
    for p in pairs:
        i = p["pair_index"]
        if i in done:
            continue
        lk_q = primary_lakota(p["lakota"])
        eng = p["english"]
        res = serper(f'"{lk_q}"', key)
        organic = res.get("organic", []) if isinstance(res, dict) else []
        lk_n, eng_n = norm(lk_q), norm(eng)

        lak_hit, hit_urls = False, []
        for o in organic:
            blob = norm(f"{o.get('title', '')} {o.get('snippet', '')}")
            if lk_n and lk_n in blob:
                lak_hit = True
                hit_urls.append(o.get("link", ""))

        eng_coloc, coloc_src = False, []
        if lak_hit:
            for u in hit_urls[:5]:
                if not u:
                    continue
                txt = fetch_page_text(u)
                if txt == "__PDF__":
                    coloc_src.append({"url": u, "status": "pdf_skipped"})
                    continue
                if txt is None:
                    coloc_src.append({"url": u, "status": "fetch_failed"})
                    continue
                tn = norm(txt)
                page_lak = lk_n in tn
                page_eng = len(eng_n) > 8 and eng_n in tn
                coloc_src.append({"url": u, "lak_on_page": page_lak, "eng_on_page": page_eng})
                if page_lak and page_eng:
                    eng_coloc = True

        rec = {"idx": i, "lakota": p["lakota"], "english": eng, "query": lk_q,
               "lakota_verbatim_hit": lak_hit, "english_colocated": eng_coloc,
               "hit_urls": hit_urls, "tier2": coloc_src,
               "serper_error": res.get("_error") if isinstance(res, dict) else None}
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fout.flush()
        n_lak += lak_hit
        n_both += eng_coloc
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(pairs)} | lak-verbatim {n_lak} | both-sided {n_both}")
        time.sleep(args.delay)
    fout.close()

    recs = [json.loads(l) for l in OUT_JSONL.read_text(encoding="utf-8").splitlines() if l.strip()]
    N = len(recs)
    lak = sum(r["lakota_verbatim_hit"] for r in recs)
    both = sum(r["english_colocated"] for r in recs)
    print("\n=== Open-web contamination (Serper, verbatim quoted Lakota) ===")
    print(f"  pairs checked:                  {N}")
    print(f"  Lakota verbatim present:        {lak}/{N} ({100 * lak / N:.0f}%)" if N else "  no pairs")
    print(f"  + English reference co-located: {both}/{N} ({100 * both / N:.0f}%)"
          "  <- aligned-pair / both-sided signal" if N else "")
    print("  (Caveats: Google index != training corpus; possible diacritic "
          "normalization; page-text proxy; PDFs flagged not deep-parsed.)")


if __name__ == "__main__":
    main()

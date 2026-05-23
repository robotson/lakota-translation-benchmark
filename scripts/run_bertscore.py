#!/usr/bin/env python3
"""Run BERTScore on lak→eng composite results and report alongside chrF++."""

import json
import os
from bert_score import score as bert_score
from pathlib import Path

COMPOSITE_DIR = Path("results/composite")
DIRECTIONS = ["lak_to_eng"]  # Only English output makes sense for BERTScore

models = sorted([d.name for d in COMPOSITE_DIR.iterdir() if d.is_dir()])

for model_dir in models:
    for direction in DIRECTIONS:
        fpath = COMPOSITE_DIR / model_dir / f"{direction}.jsonl"
        if not fpath.exists():
            continue
        
        records = []
        with open(fpath) as f:
            for line in f:
                records.append(json.loads(line))
        
        # Get model output — use translation field, fall back to raw_response
        def get_hyp(r):
            if r.get("translation") and r["translation"].strip():
                return r["translation"]
            if r.get("raw_response") and r["raw_response"].strip():
                return r["raw_response"]
            return None

        # Filter to pairs with actual output (skip refusals, parse errors, true empties)
        translated = [r for r in records if get_hyp(r) and r.get("response_type") == "translation"]

        refs = [r["reference_text"] for r in translated]
        hyps = [get_hyp(r) for r in translated]
        chrf_scores = [r["chrf_pp"] for r in translated]
        
        print(f"\n{'='*60}")
        print(f"Model: {model_dir} | Direction: {direction}")
        print(f"Pairs with translations: {len(translated)}/{len(records)}")
        
        # Run BERTScore
        P, R, F1 = bert_score(hyps, refs, lang="en", verbose=False)
        
        f1_list = F1.tolist()
        
        # Summary stats
        import statistics
        print(f"\nBERTScore F1:  mean={statistics.mean(f1_list):.3f}  median={statistics.median(f1_list):.3f}  stdev={statistics.stdev(f1_list):.3f}")
        print(f"chrF++ mean:  {statistics.mean(chrf_scores):.1f}  median: {statistics.median(chrf_scores):.1f}")
        
        # Correlation
        from scipy import stats as sp_stats
        r, p = sp_stats.pearsonr(chrf_scores, f1_list)
        print(f"Pearson r(chrF++, BERTScore): {r:.3f} (p={p:.2e})")
        
        # Most interesting: high BERTScore but low chrF++ (valid paraphrases)
        paired = list(zip(chrf_scores, f1_list, translated))
        # Sort by gap: BERTScore high relative to chrF++
        # Normalize both to 0-1 scale for comparison
        max_chrf = max(chrf_scores) if chrf_scores else 1
        gaps = [(bs - chrf/100, chrf, bs, rec) for chrf, bs, rec in paired]
        gaps.sort(key=lambda x: x[0], reverse=True)
        
        print(f"\n--- Top 10 'valid paraphrases' (high BERTScore, low chrF++) ---")
        for gap, chrf, bs, rec in gaps[:10]:
            print(f"  chrF++={chrf:5.1f}  BERTScore={bs:.3f}  gap={gap:+.3f}")
            print(f"    Ref: {rec['reference_text']}")
            print(f"    Hyp: {get_hyp(rec)}")
            print()
        
        # Save full results
        outpath = COMPOSITE_DIR / model_dir / f"{direction}_bertscore.jsonl"
        with open(outpath, 'w') as f:
            for i, rec in enumerate(translated):
                rec["bertscore_f1"] = f1_list[i]
                rec["bertscore_precision"] = P[i].item()
                rec["bertscore_recall"] = R[i].item()
                f.write(json.dumps(rec) + "\n")
        print(f"Saved: {outpath}")


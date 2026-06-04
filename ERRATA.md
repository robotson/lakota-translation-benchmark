# Errata — *Evaluating Frontier LLM Translation Capability for Lakota*

This file records corrections to the camera-ready version of the paper as
published at AmericasNLP 2026 / ACL Anthology. Each entry names the
location, the published value, the correct value, the cause, the
verification, and the impact.

---

## Table 4 (Stratified chrF++ by conversational score)

**Cell:** Gemini 3.1 Pro, English → Lakota, conversational score 6.

**Published value:** **61.3**
**Correct value:** **60.1**

**Cause:** Transcription error. The cell was filled with Gemini's
L → E thinking-condition *semantic-equivalence percentage* (61.3, which
appears three times elsewhere in the paper — in Table 8, in Section 4
prose, and in the abstract framing) instead of its computed E → L
score-6 *chrF++ mean* (60.1).

**Verification:** Recomputed from
`results/composite/gemini_thinking/eng_to_lak.jsonl` in the working-repo
canonical snapshot (commit `484efa1`, Feb–Mar 2026), n = 34 pairs at
conversational score 6, filtering response_type ∉ {refusal, empty} and
chrf_pp non-null — the same pipeline that reconciles the **other 41 of
42** Table 4 cells exactly (deviation ≤ 0.05). Neither Gemini condition
produces 61.3: baseline = **53.5**, thinking = **60.1**. The literal value
"60.1" appears nowhere in the camera-ready or in any analysis report;
"61.3" appears only as the semantic-equivalence percentage.

**Impact:** **None on any ranking, direction, or claim.** Gemini still
leads E → L score-6 substantially (next-best is Opus at 45.6, a 14.5-point
gap); the 1.2-point correction is well within per-pair variance for that
bucket. The stratification pattern (score-6 ≫ score-4 across all models
and both directions) is unaffected. No discussion or conclusion in the
paper depends on this specific cell value.

**Status:** Not eligible for camera-ready correction — the camera-ready
deadline (2026-05-22 AOE) had passed when the discrepancy was identified
during post-submission verification on 2026-06-04. To be submitted as an
ACL Anthology erratum once the proceedings are published (workshop date:
July 2026).

---

## Anthology correction procedure (action item)

After the proceedings appear on the ACL Anthology, email the Anthology
team (anthology@aclweb.org) and/or the AmericasNLP 2026 publication
chairs with:

1. A corrected PDF (regenerate `paper.tex` with line 225's Table 4 cell
   changed from `61.3` to `60.1`; the build environment is documented in
   the repo and rebuilds in seconds).
2. A short note describing the change and citing this errata entry.

The Anthology appends a revision rather than silently overwriting, so the
original record is preserved and the correction is publicly tracked.

---

## For figure-builders / future stratification work

If the stratification figure (small-multiple or slope plot of chrF++ by
conversational score 4/5/6, per model × direction) ever gets built —
**do not copy values from paper Table 4** for the Gemini E → L
score-6 cell. The paper has **61.3**; the data has **60.1**. Use
60.1, computed from the canonical per-pair snapshot. All other Table 4
cells reconcile exactly and are safe to use either way.

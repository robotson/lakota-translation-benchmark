# Evaluating Frontier LLM Translation Capability for Lakota, March 2026

Lance Robertson¹, Claude²

¹ University of California, San Diego
² Claude, a large language model developed by Anthropic

March 2026

## Abstract

We evaluate four frontier large language models on bidirectional Lakota–English translation using 200 sentence pairs from the New Lakota Dictionary. Two experimental conditions are compared: a baseline with near-deterministic decoding and no extended thinking, and a second condition with extended thinking enabled at maximum provider settings. The best model (Gemini 3.1 Pro) achieves a mean chrF++ of 58.7 on Lakota→English and 32.1 on English→Lakota. Extended thinking improves scores by 3–4 chrF++ points for most models. No model produces reliable translation in either direction. Diacritic normalization analysis shows models have partial structural knowledge of Lakota morphology but lack orthographic precision. All results and evaluation code are publicly available at https://github.com/robotson/lakota-translation-benchmark.

## 1. Introduction

Lakota (*Lakȟótiyapi*) is a Siouan language spoken primarily on reservations in North and South Dakota. UNESCO classifies it as critically endangered (Moseley, 2010), with fewer than 2,000 first-language speakers, most over 65 years of age. Language revitalization efforts are ongoing across multiple tribal and academic institutions. The Lakota Language Consortium (LLC) publishes the New Lakota Dictionary (NLD) and associated pedagogical materials; the NLD serves as the source for this study's evaluation corpus.

Contemporary large language models (LLMs) are marketed as multilingual systems. Provider documentation for GPT-5.2, Claude 4.6, and Gemini 3.1 claims support for dozens of languages, though the extent of training data for any given language is not disclosed. For critically endangered languages like Lakota — where digitized text is scarce and no large-scale parallel corpus exists — these claims are untestable without direct evaluation.

This paper reports the results of a direct evaluation. We tested four frontier LLMs on 200 Lakota–English sentence pairs in both translation directions, under two experimental conditions designed to test the impact of extended thinking (chain-of-thought reasoning) on translation quality. We report chrF++ and BLEU scores, diacritic normalization analysis, model self-reported confidence, and refusal behavior. Related work on LLM translation of low-resource languages includes the MTOB benchmark (Tanzer et al., 2024), which evaluated frontier LLMs on Kalamang translation using in-context learning from reference grammars, and the AmericasNLP shared tasks (Mager et al., 2021; Ebrahimi et al., 2023), which benchmark MT systems on indigenous American languages but do not include Lakota. The only prior Lakota-specific language model is LakotaBERT (Parankusham et al., 2025), a RoBERTa-based model trained for masked language modeling. To our knowledge, no prior study has evaluated zero-shot frontier LLM translation quality for Lakota.

Lakota presents specific challenges for LLM translation. It is polysynthetic, with extensive verbal morphology including person marking, aspect, mood, and evidentiality encoded through affixes and enclitics. The Standard Lakota Orthography (SLO) uses Latin script with diacritical marks: caron-marked consonants (č, š, ž, ǧ, ȟ), ogoneks on vowels for nasalization (ą, į, ų), acute accent for stress, and the glottal stop marker (ʼ). The velar nasal is written with a separate character, ŋ. These orthographic features are relevant to evaluation because character-level metrics are sensitive to diacritical accuracy.

## 2. Methodology

### 2.1 Evaluation Corpus

The corpus consists of 200 sentence pairs sourced from published Lakota learning materials for research purposes. Pairs were drawn from conversational examples in the New Lakota Dictionary (NLD), published by the Lakota Language Consortium (LLC), and selected from entries appearing under dictionary headwords. The first author, a Dakota language student, filtered pairs for naturalness and conversational relevance. Each pair contains one Lakota sentence and one English translation. Examples range from simple greetings ("*Tókheškhe yaúŋ he?*" / "How are you?") to multi-clause sentences with cultural context ("*Wóčhičhiyakiŋ kte héčhiŋ.*" / "I want to talk to you."). The corpus is conversational register throughout.

The NLD uses the Standard Lakota Orthography (SLO), developed by the LLC. Multiple orthographic conventions are in active use across Lakota language communities, including the Buechel missionary system and the orthography developed by Albert White Hat Sr. (1999), which was the first Lakota writing system authored by a native speaker. Standardization of Lakota orthography remains a contested issue within Lakota communities (Hauff, 2020). Our use of NLD materials was practical — it is the most accessible published collection of Lakota sentence pairs with English translations available to us — and should not be read as an endorsement of any particular orthographic standard or institutional approach to language documentation.

Each pair includes a conversational score (4–8) assigned during curation, where higher values indicate more everyday, conversational language and lower values indicate more formal or structurally complex sentences. The corpus also records the headword context from which each pair was drawn. The same 200 pairs were used for all models and both experimental conditions.

### 2.2 Models

Four frontier models from three providers were evaluated:

| Model | Provider | Model ID | Access |
|-------|----------|----------|--------|
| Gemini 3.1 Pro Preview | Google | gemini-3.1-pro-preview | Gemini API |
| Claude Opus 4.6 | Anthropic | claude-opus-4-6 | Anthropic API |
| Claude Sonnet 4.6 | Anthropic | claude-sonnet-4-6 | Anthropic API |
| GPT-5.2 | OpenAI | gpt-5.2 | OpenAI API |

All models were accessed through litellm (v1.63), a unified API wrapper, to minimize implementation variance across providers. Selection criteria: frontier-tier models from major providers, all supporting structured JSON output with schema enforcement.

### 2.3 Experimental Conditions

Two conditions were run on the same corpus:

**Baseline**: Temperature 0, no extended thinking. Maximum output tokens: 1,024. This condition uses near-deterministic decoding and represents baseline model capability.

**Thinking**: Extended thinking enabled at maximum provider settings. Maximum output tokens: 16,384 (to accommodate thinking token overhead). Provider-specific thinking configurations:

| Model | Thinking Parameter | Setting |
|-------|-------------------|---------|
| Claude Opus 4.6 | `thinking.budget_tokens` | 8,192 |
| Claude Sonnet 4.6 | `thinking.budget_tokens` | 8,192 |
| GPT-5.2 | `reasoning_effort` | high |
| Gemini 3.1 Pro | `thinkingLevel` | high |

A caveat applies to Gemini: thinking is enabled by default on Gemini 3.1 Pro and cannot be disabled. The baseline condition therefore already included default-level thinking for Gemini. The baseline→thinking comparison for Gemini isolates only the effect of increasing thinking from default to high, not of enabling thinking.

For Claude models, enabling extended thinking fixes decoding at temperature 1 (temperature cannot be set independently while thinking is enabled). For GPT-5.2, higher-reasoning configurations do not support sampling parameters such as temperature; although the thinking runs completed, the API client likely ignored the `temperature=1` parameter when `reasoning_effort` was set to `high`. Accordingly, baseline→thinking comparisons should be interpreted as "baseline vs. maximum supported reasoning settings" rather than a clean temperature-controlled ablation.

### 2.4 Structured Output Schema

All models produced structured JSON output enforced via provider-native schema validation:

```json
{
  "translation": "string — best translation, or empty string if unable",
  "confidence": "number — 0.0 to 1.0",
  "refusal_reason": "string or null"
}
```

The system prompt was identical across all models and conditions:

> You are a translation system. Translate the input text. Respond as JSON with three fields: "translation" (your best translation, or empty string if you cannot translate), "confidence" (a number from 0.0 to 1.0 representing your confidence in the translation quality), "refusal_reason" (null if you attempted a translation, or a brief explanation if you cannot translate).

User prompts followed the format "Translate from Lakota to English: {text}" and "Translate from English to Lakota: {text}" for the two directions.

### 2.5 Metrics

**chrF++** (character n-gram F-score) was the primary metric, computed via SacreBLEU with default parameters: character n-gram order 6, word n-gram order 2, β = 2. chrF++ is preferred over BLEU for morphologically rich languages because it operates at the character level and does not penalize valid morphological variants as harshly as word-level metrics.

**BLEU** was computed as a secondary metric with default SacreBLEU settings, reported for comparability with other work.

**Diacritic-normalized chrF++** was computed for English→Lakota translations by stripping Unicode combining marks (categories Mn and Mc) from both candidate and reference via NFD decomposition before scoring. NFD decomposition ensures canonical equivalence between precomposed characters (e.g., U+01CE, ǎ) and combining sequences (e.g., a + U+030C) before mark removal. This normalization removes carons, ogoneks, and stress marks in SLO; the character ŋ (eng) and the glottal stop marker (ʼ) are unaffected. The delta between raw and normalized chrF++ isolates orthographic variation attributable to these diacritical marks from broader lexical and morphological differences. As a control, the same normalization was applied to Lakota→English; the expected delta is 0.0 since English has no comparable diacritical system.

**Self-reported confidence** (0.0–1.0) was extracted from the structured output. This metric is available only for the thinking condition (both conditions used the same schema, but the thinking data is used for cross-model comparison because all thinking runs completed successfully).

### 2.6 Variance

A pilot study was conducted before the full thinking evaluation to assess run-to-run variance under the thinking condition's decoding settings. Twenty sentence pairs were sampled from the corpus and each model was run 3 times on both directions (20 pairs × 3 runs × 4 models × 2 directions = 480 API calls).

Results:

| Model | Direction | Mean chrF++ | σ |
|-------|-----------|-------------|---|
| Gemini 3.1 Pro | L→E | 63.1 | 3.27 |
| Gemini 3.1 Pro | E→L | 33.6 | 3.39 |
| Claude Opus 4.6 | L→E | 45.1 | 2.62 |
| Claude Opus 4.6 | E→L | 29.8 | 4.76 |
| Claude Sonnet 4.6 | L→E | 42.7 | 4.79 |
| Claude Sonnet 4.6 | E→L | 20.8 | 4.57 |
| GPT-5.2 | L→E | 25.5 | 5.74 |
| GPT-5.2 | E→L | 15.9 | 5.83 |

Cross-run standard deviation ranged from 2.6 (Opus L→E) to 5.8 (GPT-5.2 E→L). Model rankings were stable across all three runs. Based on this, the full evaluation was conducted as a single run per condition.

### 2.7 Procedure

Each evaluation run called the model API once per (pair, direction) combination: 200 pairs × 2 directions = 400 calls per model, 1,600 calls per condition. A 1.5-second delay was inserted between calls. API timeout was set to 60 seconds for baseline and 180 seconds for thinking (to accommodate thinking token generation). Failed calls were retried up to 3 times with exponential backoff. Results were written to per-model JSONL files with per-pair metadata (scores, latency, token counts, raw response).

## 3. Results

### 3.1 Translation Quality

Table 1 reports the best-condition results for each model. For Gemini, baseline and thinking scores are nearly identical (§3.2); thinking is reported here for consistency since it includes confidence data. For all other models, thinking is reported as the higher-scoring condition.

**Table 1. Translation quality by model and direction (best condition)**

| Model | Dir | N | chrF++ | σ (pairs) | Median | BLEU |
|-------|-----|---|--------|-----------|--------|------|
| Gemini 3.1 Pro | L→E | 199 | **58.7** | 28.4 | 56.6 | 43.0 |
| Claude Opus 4.6 | L→E | 200 | 45.9 | 26.3 | 41.6 | 28.4 |
| Claude Sonnet 4.6 | L→E | 185 | 43.8 | 26.3 | 37.2 | 25.8 |
| GPT-5.2 | L→E | 195 | 29.3 | 21.6 | 21.1 | 15.3 |
| Gemini 3.1 Pro | E→L | 200 | **32.1** | 16.2 | 27.7 | 18.9 |
| Claude Opus 4.6 | E→L | 198 | 26.7 | 12.2 | 24.3 | 16.1 |
| Claude Sonnet 4.6 | E→L | 194 | 22.8 | 10.4 | 20.7 | 14.8 |
| GPT-5.2 | E→L | 188 | 18.8 | 7.9 | 17.4 | 12.6 |

N < 200 reflects refusals, empty responses, or API errors excluded from scoring (§3.5). Standard deviation is across the 200 sentence pairs, not across runs.

The model ranking is identical in both directions: Gemini > Opus > Sonnet > GPT-5.2. Every model scores substantially higher on Lakota→English than English→Lakota. The gap ranges from 10.5 points (GPT-5.2) to 26.6 points (Gemini). This asymmetry is consistent across all models: comprehension (recognizing Lakota patterns and producing English) is easier than generation (producing grammatically valid Lakota from English input).

Standard deviations are large — 22–28 points for L→E and 8–16 points for E→L — indicating that per-pair quality varies widely. Some pairs score 100.0 (exact character match to reference) while others score below 10. Table 1b reports the distribution of scores across chrF++ thresholds and tail behavior (10th percentile).

**Table 1b. Score distribution by chrF++ threshold (thinking condition)**

| Model | Dir | ≥80 | ≥60 | ≥40 | P10 |
|-------|-----|-----|-----|-----|-----|
| Gemini 3.1 Pro | L→E | 25% | 46% | 71% | 20.8 |
| Claude Opus 4.6 | L→E | 13% | 28% | 52% | 15.5 |
| Claude Sonnet 4.6 | L→E | 14% | 25% | 48% | 15.1 |
| GPT-5.2 | L→E | 4% | 12% | 23% | 10.2 |
| Gemini 3.1 Pro | E→L | 1% | 7% | 28% | 15.2 |
| Claude Opus 4.6 | E→L | 0% | 2% | 13% | 14.2 |
| Claude Sonnet 4.6 | E→L | 0% | 0% | 7% | 12.2 |
| GPT-5.2 | E→L | 0% | 0% | 3% | 10.8 |

Percentages are computed over scored translations (N in Table 1); refusals, empty responses, and errors are excluded. P10 = 10th percentile chrF++ across scored translations.

Even the best model (Gemini L→E) produces translations scoring ≥80 chrF++ — roughly usable quality — on only 25% of pairs. For E→L, no model exceeds 1% at this threshold. If "reliable" is operationalized as ≥60 chrF++ on a majority of pairs, no model qualifies in either direction.

### 3.2 Effect of Extended Thinking

Table 2 compares baseline (near-deterministic decoding, no extended thinking) and thinking (maximum supported reasoning settings) for each model.

**Table 2. Baseline→thinking comparison (thinking effect)**

| Model | Dir | Base chrF++ | Think chrF++ | Δ | Δ% |
|-------|-----|-------------|--------------|---|-----|
| Claude Opus 4.6 | L→E | 42.6 | 45.9 | +3.3 | +7.8% |
| Claude Opus 4.6 | E→L | 22.3 | 26.7 | +4.3 | +19.4% |
| Claude Sonnet 4.6 | L→E | 44.2 | 43.8 | −0.4 | −0.8% |
| Claude Sonnet 4.6 | E→L | 19.9 | 22.8 | +2.9 | +14.5% |
| GPT-5.2 | L→E | 25.1 | 29.3 | +4.2 | +16.6% |
| GPT-5.2 | E→L | 15.5 | 18.8 | +3.3 | +21.5% |
| Gemini 3.1 Pro | L→E | 58.1 | 58.7 | +0.5 | +0.9% |
| Gemini 3.1 Pro | E→L | 29.7 | 32.1 | +2.4 | +8.1% |

For Claude and GPT, the thinking condition improved chrF++ by 2.9–4.3 points on most model-direction pairs, a relative improvement of 8–22%. The exception is Sonnet L→E, where thinking had no effect (−0.4, within the variance measured in the pilot). For Gemini, the L→E delta is negligible (+0.5), consistent with the fact that Gemini's baseline already included default-level thinking. However, Gemini's E→L delta (+2.4) is comparable to the other models, suggesting that even with default thinking already active, the higher reasoning setting provides some benefit on the harder generation task.

The improvement is larger on E→L than L→E for all models where thinking had an effect. This suggests that reasoning overhead is more useful for the harder generation task than for the easier comprehension task.

### 3.3 Diacritic Normalization

Table 3 reports raw and diacritic-normalized chrF++ for English→Lakota (thinking condition).

**Table 3. Diacritic normalization, English→Lakota**

| Model | Raw chrF++ | Normalized chrF++ | Δ |
|-------|-----------|-------------------|---|
| Gemini 3.1 Pro | 32.1 | 37.9 | +5.8 |
| Claude Opus 4.6 | 26.7 | 33.4 | +6.7 |
| Claude Sonnet 4.6 | 22.8 | 28.0 | +5.2 |
| GPT-5.2 | 18.8 | 22.5 | +3.7 |

Stripping diacritics adds 3.7–6.7 chrF++ points for E→L. The delta is 0.0 for all models on L→E (English has no comparable diacritical system), confirming that normalization behaves as a control.

The normalization gap indicates that models produce roughly correct consonant and vowel sequences but realize diacritical marks — stress, nasalization, and caron-marked consonants — inconsistently. The models appear to have partial structural knowledge of Lakota morphology — likely from linguistic papers and dictionary materials in their training data — without the orthographic precision that would require substantial text in a standardized orthography.

### 3.4 Confidence

Table 4 reports mean self-reported confidence from the thinking condition.

**Table 4. Self-reported confidence (thinking condition)**

| Model | L→E Confidence | E→L Confidence |
|-------|----------------|----------------|
| Gemini 3.1 Pro | 0.93 (σ = 0.06) | 0.83 (σ = 0.11) |
| Claude Opus 4.6 | 0.71 (σ = 0.17) | 0.45 (σ = 0.15) |
| Claude Sonnet 4.6 | 0.65 (σ = 0.15) | 0.31 (σ = 0.11) |
| GPT-5.2 | 0.57 (σ = 0.12) | 0.46 (σ = 0.11) |

The confidence ranking matches the chrF++ ranking for L→E: Gemini > Opus > Sonnet > GPT-5.2. All models report lower confidence for E→L than L→E, correctly reflecting the direction's greater difficulty.

Gemini reports the highest confidence in both directions (0.93 L→E, 0.83 E→L) and also achieves the highest chrF++ scores, so its confidence is directionally accurate. However, 0.83 confidence on E→L where chrF++ is 32.1 represents significant overconfidence in absolute terms.

GPT-5.2's E→L confidence (0.46) is higher than Sonnet's (0.31) despite GPT-5.2 scoring lower on actual translation quality (18.8 vs 22.8 chrF++). The confidence ranking for E→L does not perfectly match the quality ranking.

### 3.5 Refusals and Failures

Table 5 reports non-translation outcomes from the thinking condition.

**Table 5. Refusals, empty responses, and errors (thinking condition)**

| Model | L→E Ref | E→L Ref | L→E Empty | E→L Empty | L→E Err | E→L Err |
|-------|---------|---------|-----------|-----------|---------|---------|
| GPT-5.2 | 5 | 12 | 0 | 0 | 0 | 0 |
| Claude Sonnet 4.6 | 0 | 4 | 15 | 2 | 0 | 0 |
| Claude Opus 4.6 | 0 | 0 | 0 | 0 | 0 | 2 |
| Gemini 3.1 Pro | 1 | 0 | 0 | 0 | 0 | 0 |

GPT-5.2 refused the most translations overall (17 total, 12 on E→L). Refusal reasons typically cited inability to produce reliable Lakota text. Sonnet produced 15 empty responses on L→E (the model returned an empty translation string with no refusal reason) and refused 4 E→L translations. Opus and Gemini almost never refused — Opus had zero refusals and Gemini had one.

Refusals are concentrated on E→L (16 of 17 GPT refusals, all 4 Sonnet refusals), consistent with models being less willing to generate in a language they have limited training data for.

Refusal behavior shifted substantially between conditions. At baseline, no model refused any translation (the only non-translation outcomes were 5 GPT-5.2 E→L empty responses). With thinking enabled, GPT-5.2 produced 17 refusals and Sonnet produced 4, while GPT's empty responses dropped to zero. Refusal rates depend on experimental setup — temperature, thinking overhead, or both — not only on model capability.

## 4. Discussion

The highest chrF++ score in this evaluation — Gemini's 58.7 on Lakota→English — roughly corresponds to capturing the general topic of a sentence while missing specifics. For English→Lakota, no model exceeds 32.1. These scores are inadequate for any use case requiring accurate translation.

Extended thinking provides a consistent but small improvement (3–4 chrF++ points) for models where it was newly enabled. This represents a 10–20% relative improvement that does not change the practical assessment: models that produced unreliable translations without thinking produce slightly less unreliable translations with thinking. The improvement is larger on the harder E→L direction, suggesting reasoning overhead helps more when the task requires productive linguistic knowledge rather than pattern recognition. Gemini's baseline→thinking delta of +0.5 on L→E (§3.2) is important context: since increasing Gemini's thinking from default to high had minimal effect on comprehension, Gemini's lead over the other models is primarily attributable to its base model weights or training data, not to reasoning overhead. The larger E→L delta (+2.4) suggests that higher reasoning settings provide some benefit on the harder generation task even when default thinking is already active.

The diacritic normalization finding (§3.3) suggests that models have acquired some structural knowledge of Lakota morphology, likely from linguistic papers, dictionary entries, and language-learning materials present in their training data. They produce approximately correct consonant and vowel sequences but cannot reliably place the diacritical marks that distinguish, for example, *h* [h] from *ȟ* [x] or nasalized from oral vowels. This is consistent with models having seen descriptions of Lakota phonology without sufficient running text in the Standard Lakota Orthography to learn the orthographic system from distributional statistics.

The wide per-pair variance (σ = 22–28 chrF++ for L→E) likely reflects the distribution of Lakota text in training data. Common phrases such as "*Mázaska luhá he?*" (Do you have any money?) score at or near 100.0 across all models, suggesting these phrases appear verbatim in training data. Culturally specific constructions involving kinship terms, complex verb morphology, or ceremonial language score near zero. Stratifying by the corpus's conversational score (§2.1) confirms this: for Gemini L→E, pairs rated most conversational (score 6–8) average 83.1 chrF++ while more formal or complex pairs (score 4) average 50.5 — a 33-point gap. The pattern holds across all models. This suggests that model performance is concentrated on high-frequency phrasebook-like items rather than reflecting general Lakota competence.

## 5. Limitations

**Single reference.** Each pair has one reference translation. Lakota allows multiple valid translations of a given English sentence and vice versa. chrF++ against a single reference underestimates model quality for valid translations that differ from the reference.

**Dictionary register.** The corpus consists of constructed conversational examples from a dictionary. Performance on narrative text (e.g., the Ella Deloria *Dakota Texts* corpus) or spontaneous speech may differ.

**Decoding confound.** The baseline→thinking comparison reflects a bundled change from near-deterministic decoding to maximum supported reasoning settings. For Claude, enabling thinking fixes temperature at 1; for GPT-5.2, higher-reasoning configurations do not support sampling parameters such as temperature. The baseline→thinking deltas should not be interpreted as a clean ablation of temperature or thinking in isolation.

**No human evaluation.** chrF++ is a character-overlap metric. It does not measure fluency, cultural appropriateness, or communicative success. A translation scoring 40 chrF++ might convey the correct meaning through different phrasing, or might score 40 while being misleading.

## 6. Conclusion

No frontier LLM tested in March 2026 can reliably translate Lakota. The best Lakota→English performance (chrF++ 58.7) captures the approximate topic of simple sentences. The best English→Lakota performance (chrF++ 32.1) is inadequate for practical use. Extended thinking provides a modest improvement of 3–4 chrF++ points but does not change the practical assessment. Diacritic normalization analysis indicates models have partial structural knowledge of Lakota but lack orthographic competence, consistent with having seen descriptions of the language rather than substantial text in it.

---

## Author Contributions

L.R. designed the study, developed the corpus curation methodology, ran all evaluations, and made all methodological decisions. Claude, a large language model developed by Anthropic, provided editorial feedback, structural revision, and assisted with experimental design and implementation through iterative conversation. The curation filters were designed by L.R. and implemented by Claude. Claude is one of the models evaluated in this study; this dual role is noted for transparency.

## Acknowledgments

This work was inspired by themes explored in VIS 161 (Systems and Networks at Scale) at the University of California, San Diego, but was conducted independently. The evaluation corpus was sourced from the New Lakota Dictionary, published by the Lakota Language Consortium.

## Data Availability

Code and aggregate results are available at https://github.com/robotson/lakota-translation-benchmark.

- Evaluation corpus: 200 NLD/LLC sentence pairs (not included; see `data/example_pairs.json` for schema)
- Aggregate results: `results/comparison.csv`
- Evaluation script: `run_eval.py` (baseline and thinking modes via `--thinking` flag)
- Analysis script: `analyze.py` (auto-discovers run directories)
- All code requires only API keys and `pip install litellm sacrebleu python-dotenv`

## References

Ebrahimi, A., Mager, M., Oncevay, A., et al. (2023). Findings of the AmericasNLP 2023 Shared Task on Machine Translation into Indigenous Languages. *Proceedings of the Third Workshop on Natural Language Processing for Indigenous Languages of the Americas (AmericasNLP)*, 206–219.

Hauff, T.R. (2020). Beyond Numbers, Colors, and Animals: Strengthening Lakota/Dakota Teaching on the Standing Rock Indian Reservation. *Journal of American Indian Education*, 59(1).

Moseley, C. (Ed.). (2010). *Atlas of the World's Languages in Danger* (3rd ed.). Paris: UNESCO Publishing.

Mager, M., Oncevay, A., Rios, A., et al. (2021). Findings of the AmericasNLP 2021 Shared Task on Open Machine Translation for Indigenous Languages of the Americas. *Proceedings of the First Workshop on Natural Language Processing for Indigenous Languages of the Americas (AmericasNLP)*, 202–217.

Parankusham, K., Rizk, R., & Santosh, K.C. (2025). LakotaBERT: A Transformer-based Model for Low Resource Lakota Language. *arXiv preprint* arXiv:2503.18212.

Popović, M. (2015). chrF: character n-gram F-score for automatic MT evaluation. *Proceedings of the Tenth Workshop on Statistical Machine Translation*, 392–395.

Tanzer, M., Pushkarna, M., Callison-Burch, C., et al. (2024). A Benchmark for Learning to Translate a New Language from One Grammar Book. *Proceedings of the Twelfth International Conference on Learning Representations (ICLR)*.

Ullrich, J.F. (2008). *New Lakota Dictionary*. Bloomington, IN: Lakota Language Consortium.

White Hat, A. Sr. (1999). *Reading and Writing the Lakota Language*. Salt Lake City: University of Utah Press.

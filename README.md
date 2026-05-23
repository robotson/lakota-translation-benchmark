# Lakota LLM Translation Evaluation

Benchmarking frontier and open-weight LLMs on bidirectional Lakota–English translation.

## What This Is

A reproducible evaluation of **seven large language models** — four proprietary
(Claude Opus 4.6, Claude Sonnet 4.6, GPT-5.2, Gemini 3.1 Pro) and three
open-weight (DeepSeek-V4-Pro, GLM-5.1, Qwen3.6-Plus, via Together AI) — on 200
Lakota–English sentence pairs, tested in both directions and, for each model,
with and without extended reasoning. Outputs are scored with chrF++ and BLEU
(SacreBLEU), with a diacritic-normalization analysis, and Lakota→English outputs
are additionally judged for semantic equivalence by **two independent LLM judges**
from different vendors (Gemini 3 Flash and Llama-3.3-70B) to measure inter-judge
agreement. Open-web overlap is audited to bound data-contamination risk.

See [paper.pdf](paper.pdf) for the full writeup (AmericasNLP 2026).

## Key Findings

- **No model produces reliable Lakota translation**, proprietary or open-weight.
  Best Lakota→English chrF++ is 59.4 (Gemini); best English→Lakota is 42.6.
- **chrF++ overstates comprehension.** An LLM semantic judge finds Gemini reaches
  60.4% semantic equivalence on L→E while GPT-5.2 reaches only 6% — despite both
  producing fluent English. The chrF++↔BERTScore correlation tracks this and acts
  as a lightweight hallucination signal.
- **Open-weight models do not close the gap.** The strongest open model
  (DeepSeek-V4-Pro, 38.0 / 29.2 chrF++) lands between GPT-5.2 and the Claude
  models; GLM-5.1 and Qwen3.6-Plus sit at GPT-5.2's baseline tier. None approach
  Gemini.
- **For open-weight models, reasoning changes refusal behavior more than quality.**
  Enabling reasoning leaves their semantic equivalence essentially flat but sharply
  changes refusals — it surfaces the recognition that the model cannot translate
  Lakota rather than improving the output.
- **Two judges agree substantially** (Cohen's κ = 0.75 over 2,758 pairs), so the
  capability ranking is not an artifact of a single judge or judge family.
- **Diacritic inconsistency** — models get roughly the right base characters but
  place diacritical marks inconsistently, possibly reflecting orthographic
  heterogeneity in training data.
- **Contamination is bounded.** A verbatim open-web check finds the Lakota source
  online for ~10% of pairs but the aligned English reference co-located for only
  0.5%; scores span a wide range rather than saturating, inconsistent with
  effective test-set memorization.

## Quick Start

```bash
git clone https://github.com/robotson/lakota-translation-benchmark.git
cd lakota-translation-benchmark
pip install -r requirements.txt
cp .env.example .env
# Add your API keys to .env
```

### Add Your Data

The evaluation corpus is not included — it is community language data that we
prefer to keep off the open web. Supply your own sentence pairs in the format
shown in `data/example_pairs.json`:

```json
{
  "source": "your-source-name",
  "register": "conversational",
  "pairs": [
    {
      "lakota": "Háu, tókheškhe yaúŋ he?",
      "english": "Hello, how are you?",
      "context": "greeting"
    }
  ]
}
```

Place your JSON file(s) in `data/holdout/` (create the directory). The eval
scripts load all `.json` files from that directory.

### Run Evaluations

```bash
# Proprietary models (Anthropic / OpenAI / Google)
python scripts/run_eval.py --dry-run                  # preview
python scripts/run_eval.py                            # baseline (temp 0)
python scripts/run_eval.py --thinking                 # extended reasoning

# Open-weight models (DeepSeek / GLM / Qwen via Together)
python scripts/run_eval_oss.py --dry-run
python scripts/run_eval_oss.py --mode thinking        # reasoning on
python scripts/run_eval_oss.py --mode nonthinking     # reasoning disabled (ablation)
```

### Judge, Agreement, Contamination

```bash
# Two independent semantic-equivalence judges over L→E composite results
python scripts/run_llm_judge.py                       # Gemini 3 Flash
python scripts/run_llm_judge_llama.py                 # Llama-3.3-70B (Together)

# Join the two judge passes and compute inter-judge agreement
python scripts/collate_judgments.py                   # -> judgments_joined.csv
python scripts/compute_interjudge_kappa.py            # Cohen's / weighted kappa

# Open-web overlap audit (requires SERPER_API_KEY)
python scripts/check_openweb_contamination.py
```

## Scripts

| Script | Description |
|--------|-------------|
| `run_eval.py` | Proprietary-model translation eval. `--thinking` toggles extended reasoning; `--pilot` runs the variance pilot |
| `run_eval_oss.py` | Open-weight eval (DeepSeek/GLM/Qwen via Together). Imports the exact scoring contract from `run_eval`. `--mode {thinking,nonthinking}` is a clean within-model reasoning ablation |
| `analyze.py` | Compares baseline and thinking results; produces comparison tables and summary CSV |
| `run_bertscore.py` | Computes BERTScore (roberta-large) on L→E composite results |
| `run_llm_judge.py` | Primary LLM semantic judge (Gemini 3 Flash) — rates each L→E hypothesis–reference pair equivalent / partially\_equivalent / not\_equivalent. Resumable JSONL |
| `run_llm_judge_llama.py` | Independent second judge (Llama-3.3-70B via Together), contract-identical, separate output file |
| `collate_judgments.py` | Joins the two judge passes into `judgments_joined.csv` |
| `compute_interjudge_kappa.py` | Cohen's and quadratic-weighted κ between the two judges; per-model and proprietary-vs-open-weight splits |
| `check_openweb_contamination.py` | Verbatim open-web overlap audit via Serper (SSRF-guarded page fetches) |

## Results

`results/comparison.csv` contains aggregate statistics for all seven models
(both directions × baseline/thinking). Columns:

| Column | Description |
|--------|-------------|
| `model` | Model name |
| `direction` | L→E (Lakota→English) or E→L (English→Lakota) |
| `mode` | `baseline` (no/disabled reasoning) or `thinking` (reasoning on) |
| `n_translations` | Successful translation count |
| `n_refusals` / `n_empties` / `n_errors` | Non-translation response counts |
| `chrf_mean` / `chrf_median` / `chrf_stdev` | chrF++ sentence-level statistics |
| `bleu_mean` | BLEU score |
| `conf_mean` | Model self-reported confidence |

`results/llm_judge_summary.csv` contains aggregate semantic-judge results for the
L→E direction across all 14 model×condition combinations (Gemini 3 Flash judge);
inter-judge agreement with the second judge is computed by
`compute_interjudge_kappa.py`.

## Models Tested

| Model | Provider | Reasoning toggle |
|-------|----------|------------------|
| Claude Opus 4.6 | Anthropic | `budget_tokens` extended thinking |
| Claude Sonnet 4.6 | Anthropic | `budget_tokens` extended thinking |
| GPT-5.2 | OpenAI | `reasoning_effort=high` |
| Gemini 3.1 Pro | Google | `thinkingLevel=high` (on by default) |
| DeepSeek-V4-Pro | Together | reasoning on by default; off via `reasoning.enabled=false` |
| GLM-5.1 | Together | reasoning off via `chat_template_kwargs.enable_thinking=false` |
| Qwen3.6-Plus | Together | reasoning off via `chat_template_kwargs.enable_thinking=false` |

Evaluation dates (API model versions current on these dates): proprietary models
February–March 2026; open-weight models May 2026. Exact model strings are in
`scripts/run_eval.py` (`DEFAULT_MODELS`) and `scripts/run_eval_oss.py` (`MODELS`).

## Citation

```bibtex
@inproceedings{robertson2026lakota,
  title={Evaluating Frontier LLM Translation Capability for Lakota},
  author={Robertson, Lance},
  booktitle={Proceedings of the Workshop on NLP for Indigenous Languages of the Americas (AmericasNLP)},
  year={2026}
}
```

## License

MIT

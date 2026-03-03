# Lakota LLM Translation Evaluation

Benchmarking frontier LLMs on bidirectional Lakota–English translation.

## What This Is

A reproducible evaluation of four frontier LLMs (Claude Opus 4.6, Claude Sonnet 4.6, GPT-5.2, Gemini 3.1 Pro) on 200 Lakota–English sentence pairs, tested in both translation directions under two conditions: baseline (temperature 0, no thinking) and extended thinking (maximum reasoning settings per provider). Scored with chrF++ and BLEU via SacreBLEU, with diacritic normalization analysis.

See [PAPER.md](PAPER.md) for the full writeup.

## Key Findings (March 2026)

- **No model produces reliable Lakota translation.** The best L→E score is chrF++ 58.7 (Gemini); E→L drops to 18–33 across all models.
- **Extended thinking helps modestly** — +3–4 chrF++ points for most models, nowhere near enough to cross from unreliable to usable.
- **English→Lakota is dramatically harder** — every model scores 15–30 points lower, reflecting the difficulty of generating valid Lakota morphology.
- **Diacritic normalization reveals hidden competence** — models get roughly the right consonants and morphemes but misplace stress marks, nasalization, and caron-marked consonants.

## Quick Start

```bash
git clone <this-repo>
cd lakota-llm-eval
pip install -r requirements.txt
cp .env.example .env
# Add your API keys to .env
```

### Add Your Data

The evaluation corpus is not included (it was sourced from copyrighted published materials). You need to supply your own sentence pairs in the format shown in `data/example_pairs.json`:

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

Place your JSON file(s) in `data/holdout/` (create the directory). The eval scripts load all `.json` files from that directory.

### Run Evaluations

```bash
# Baseline: temp=0, no extended thinking
python scripts/run_eval.py --dry-run                  # preview
python scripts/run_eval.py --sample 5                 # test on 5 pairs
python scripts/run_eval.py                            # full run

# Thinking: temp=1, extended reasoning enabled
python scripts/run_eval.py --thinking --pilot         # 20-pair variance pilot
python scripts/run_eval.py --thinking                 # full run
python scripts/run_eval.py --thinking --models claude-opus-4-6  # single model
```

Results go to `results/baseline/` or `results/thinking/` automatically.

### Analyze Results

```bash
python scripts/analyze.py                             # auto-discovers runs
python scripts/analyze.py --csv-out results/out.csv   # custom CSV path
```

## Scripts

| Script | Description |
|--------|-------------|
| `run_eval.py` | Runs the evaluation. `--thinking` toggles extended reasoning (changes temp, token limits, and enables per-model thinking config). Includes `--pilot` mode for variance analysis and `--runs N` for repeated evaluation |
| `analyze.py` | Compares baseline and thinking results. Auto-discovers run directories, produces comparison tables and summary CSV |

## Results

`results/comparison.csv` contains aggregate statistics from our March 2026 evaluation. Columns:

| Column | Description |
|--------|-------------|
| `model` | Model name |
| `direction` | L→E (Lakota→English) or E→L (English→Lakota) |
| `mode` | `baseline` or `thinking` |
| `n_translations` | Successful translation count |
| `n_refusals` / `n_empties` / `n_errors` | Non-translation response counts |
| `chrf_mean` / `chrf_median` / `chrf_stdev` | chrF++ sentence-level statistics |
| `norm_chrf_mean` | Diacritic-normalized chrF++ (E→L only) |
| `bleu_mean` | BLEU score |
| `conf_mean` / `conf_stdev` | Model self-reported confidence (thinking mode only) |

## Models Tested

| Model | Provider | Thinking Config |
|-------|----------|----------------|
| Claude Opus 4.6 | Anthropic | `budget_tokens=8192` |
| Claude Sonnet 4.6 | Anthropic | `budget_tokens=8192` |
| GPT-5.2 | OpenAI | `reasoning_effort=high` |
| Gemini 3.1 Pro | Google | `thinkingLevel=high` (on by default) |

## Citation

```bibtex
@misc{robertson2026lakota,
  title={Evaluating Frontier LLM Translation Capability for Lakota},
  author={Robertson, Lance},
  year={2026},
  month={March},
  howpublished={\url{https://github.com/TODO}}
}
```

## License

MIT

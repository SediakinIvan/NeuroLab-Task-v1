# Agent-Based Social Survey Analysis (GigaChat)

Pipeline for social survey analysis with:

- structured EDA and outlier analysis
- Russian text preprocessing + embeddings + clustering
- LLM-based cluster interpretation and final report generation
- merge-stage statistical relationship analysis

## Quick start

1. Install dependencies:
   - `python -m pip install -r requirements.txt`
2. Configure secrets:
   - copy `.env.example` to `.env`
   - set `GIGACHAT_API_KEY`
3. Run:
   - full pipeline: `python -m src.main --config configs/default.yaml --profile full`
   - fast local pass: `python -m src.main --config configs/default.yaml --profile quick`
   - no external LLM calls: `python -m src.main --config configs/default.yaml --profile no-llm`

## Run profiles

- `full`: all enabled stages
- `quick`: disables embeddings/clustering/merge/LLM report stages for fast checks
- `no-llm`: keeps analytics stages, disables external GigaChat calls (fallback reports only)

## Outputs

- `data/interim/`: split datasets
- `data/processed/`: merged final dataset
- `artifacts/plots/`: EDA plots
- `artifacts/reports/`: JSON/Markdown reports
- `artifacts/logs/run_metadata.json`: runtime metadata (timestamp, config hash, runtime, errors)

## Example Result

- Commit-friendly run snapshot: `example_result/EXAMPLE_RESULT.md`

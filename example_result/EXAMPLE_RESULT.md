# Example Result Snapshot

This file captures a representative successful run of the pipeline on `Students Social Media Addiction.csv`.

## Run Input

- Rows: `200`
- Columns: `15`
- Primary ID column: `Student_ID`
- Language in free-text fields: Russian

## Text Preprocessing

- Non-empty cleaned rows: `200`
- Russian rows: `200`
- Non-Russian rows: `0`
- Mean Russian character ratio: `0.9136`

## Embeddings

- Model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Rows embedded: `200`
- Embedding dimension: `384`
- Batch size: `32`
- Normalized embeddings: `true`

## Clustering

- Algorithm: `kmeans`
- Selected clusters: `12`
- Silhouette score: `0.2717`
- Davies-Bouldin score: `1.5934`
- Quality status: `ok`

Cluster sizes:

- `0: 17`, `1: 9`, `2: 14`, `3: 32`, `4: 6`, `5: 19`, `6: 6`, `7: 22`, `8: 30`, `9: 12`, `10: 14`, `11: 19`

## Merge and Statistical Analysis

- Structured rows: `200`
- Text rows: `200`
- Merged rows: `200`
- Dropped rows from structured/text: `0 / 0`

Strong categorical associations with `cluster_id`:

- `Gender`: Cramer's V = `0.8369`
- `Country`: Cramer's V = `0.8316`
- `Affects_Academic_Performance`: Cramer's V = `0.7969`
- `Most_Used_Platform`: Cramer's V = `0.7040`

Strong numeric differences across clusters:

- `Addicted_Score`: eta^2 = `0.6679`
- `Avg_Daily_Usage_Hours`: eta^2 = `0.6616`
- `Mental_Health_Score`: eta^2 = `0.6407`
- `Conflicts_Over_Social_Media`: eta^2 = `0.6279`

## Final LLM Report

- Source: `gigachat`
- Language: `ru`
- Title: `Анализ данных о зависимости от социальных сетей и их влиянии на здоровье`

## Main Output Files Produced During the Run

- `data/interim/structured.parquet`
- `data/interim/text.parquet`
- `data/processed/final_dataset.parquet`
- `artifacts/reports/final_report.md`
- `artifacts/reports/final_report.json`

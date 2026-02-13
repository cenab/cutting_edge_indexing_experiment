# Indexing Experiment Framework

This package implements all 39 indexing methods listed in `WHAT_TO_IMPLEMENT.md` and exposes a unified API for build/search/evaluation.

## Quick start

1. Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

2. Run one method:

```bash
python indexing_experiments/framework.py \
  --method 1.1_fixed_length_chunking \
  --corpus data/sample_corpus.jsonl \
  --queries data/sample_queries.jsonl \
  --output-dir results
```

3. Run all methods:

```bash
python indexing_experiments/framework.py \
  --all \
  --corpus data/sample_corpus.jsonl \
  --queries data/sample_queries.jsonl \
  --output-dir results
```

4. Generate notebooks:

```bash
python scripts/generate_notebooks.py
```

Generated notebooks are written to `indexing_notebooks/`.

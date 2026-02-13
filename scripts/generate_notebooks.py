from __future__ import annotations

import json
from pathlib import Path

from indexing_experiments import METHOD_SPECS, method_slug


def markdown_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().splitlines()],
    }


def code_cell(code: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code.strip().splitlines()],
    }


def notebook_object(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def make_method_notebook(method_id: str, category: str, name: str, description: str) -> dict:
    md = f"""
# {name}

- `method_id`: `{method_id}`
- `category`: `{category}`
- `goal`: {description}

This notebook runs one method with a shared evaluation pipeline and saves results to `results/{method_slug(method_id)}.json`.
"""

    setup_code = """
from pathlib import Path
import sys

repo_root = Path.cwd()
if not (repo_root / "indexing_experiments").exists():
    repo_root = repo_root.parent

if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from indexing_experiments import run_method
"""

    run_code = f"""
METHOD_ID = "{method_id}"

result = run_method(
    method_id=METHOD_ID,
    corpus_path=repo_root / "data" / "sample_corpus.jsonl",
    queries_path=repo_root / "data" / "sample_queries.jsonl",
    output_dir=repo_root / "results",
    top_k=5,
)

result["metrics"]
"""

    inspect_code = """
import pandas as pd

df = pd.DataFrame(result["per_query"])
df[["query", "precision_at_k", "recall_at_k", "mrr", "ndcg_at_k"]]
"""

    path_code = """
print("Saved:", result["result_path"])
"""

    return notebook_object(
        [
            markdown_cell(md),
            code_cell(setup_code),
            code_cell(run_code),
            code_cell(inspect_code),
            code_cell(path_code),
        ]
    )


def make_comparison_notebook() -> dict:
    md = """
# Indexing Method Comparison

This notebook loads all method result files from `results/`, builds a ranking table, and shows category-level summaries.
"""

    setup_code = """
from pathlib import Path
import json
import sys
import pandas as pd

repo_root = Path.cwd()
if not (repo_root / "indexing_experiments").exists():
    repo_root = repo_root.parent

if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from indexing_experiments import METHOD_SPECS
"""

    load_code = """
results_dir = repo_root / "results"
rows = []
for path in sorted(results_dir.glob("*.json")):
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics", {})
    rows.append(
        {
            "method_id": payload.get("method_id"),
            "method_name": payload.get("method_name"),
            "category": payload.get("category"),
            "precision_at_k": metrics.get("precision_at_k", 0.0),
            "recall_at_k": metrics.get("recall_at_k", 0.0),
            "mrr": metrics.get("mrr", 0.0),
            "ndcg_at_k": metrics.get("ndcg_at_k", 0.0),
            "chunks": payload.get("chunks", 0),
            "result_file": path.name,
        }
    )

score_df = pd.DataFrame(rows)
score_df.sort_values(["mrr", "ndcg_at_k"], ascending=False).reset_index(drop=True)
"""

    category_code = """
if len(score_df) == 0:
    print("No results found in results/. Run method notebooks first.")
else:
    category_summary = (
        score_df.groupby("category", as_index=False)[["precision_at_k", "recall_at_k", "mrr", "ndcg_at_k"]]
        .mean()
        .sort_values("mrr", ascending=False)
    )
    display(category_summary)
"""

    coverage_code = """
expected = {spec.method_id for spec in METHOD_SPECS}
available = set(score_df["method_id"].tolist()) if len(score_df) else set()
missing = sorted(expected - available)

print(f"Expected methods: {len(expected)}")
print(f"Available result files: {len(available)}")
print("Missing methods:")
missing
"""

    return notebook_object(
        [
            markdown_cell(md),
            code_cell(setup_code),
            code_cell(load_code),
            code_cell(category_code),
            code_cell(coverage_code),
        ]
    )


def write_notebook(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    out_dir = repo_root / "indexing_notebooks"

    generated = []
    for spec in METHOD_SPECS:
        filename = f"{method_slug(spec.method_id)}_{spec.name.lower().replace(' ', '_')}.ipynb"
        filename = filename.replace("-", "_")
        nb_path = out_dir / filename
        nb = make_method_notebook(spec.method_id, spec.category, spec.name, spec.description)
        write_notebook(nb_path, nb)
        generated.append(nb_path)

    compare_path = out_dir / "compare_indexing_methods.ipynb"
    write_notebook(compare_path, make_comparison_notebook())

    print(f"Generated method notebooks: {len(generated)}")
    print(f"Generated comparison notebook: {compare_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from pathlib import Path

from indexing_experiments.llama_framework import METHOD_SPECS, method_slug


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
        "source": [line + "\n" for line in code.strip("\n").splitlines()],
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


def load_core_source(repo_root: Path) -> str:
    source_path = repo_root / "indexing_experiments" / "llama_framework.py"
    source = source_path.read_text(encoding="utf-8")

    marker = "\ndef _main() -> None:\n"
    idx = source.find(marker)
    if idx != -1:
        source = source[:idx].rstrip() + "\n"

    return source


def make_method_notebook(
    method_id: str,
    category: str,
    name: str,
    description: str,
    core_source: str,
) -> dict:
    md = f"""
# {name} (LlamaIndex, Isolated)

- `method_id`: `{method_id}`
- `category`: `{category}`
- `goal`: {description}

This notebook is fully self-contained. It includes the full experiment implementation and does not import project-local framework modules at runtime.
"""

    install_code = """
# Run this cell in a fresh notebook environment.
%pip install -q llama-index llama-index-embeddings-huggingface llama-index-retrievers-bm25 numpy pandas scikit-learn networkx
"""

    setup_code = """
from pathlib import Path
import pandas as pd

repo_root = Path.cwd()
if not (repo_root / "data").exists():
    repo_root = repo_root.parent

corpus_path = repo_root / "data" / "sample_corpus.jsonl"
queries_path = repo_root / "data" / "sample_queries.jsonl"
results_dir = repo_root / "results"
results_dir.mkdir(parents=True, exist_ok=True)
"""

    run_code = f"""
METHOD_ID = "{method_id}"
TOP_K = 5

result = run_method(
    method_id=METHOD_ID,
    corpus_path=corpus_path,
    queries_path=queries_path,
    output_dir=results_dir,
    top_k=TOP_K,
)

result["metrics"]
"""

    inspect_code = """
df = pd.DataFrame(result["per_query"])
df[["query", "precision_at_k", "recall_at_k", "mrr", "ndcg_at_k"]]
"""

    save_code = """
print("Saved:", result["result_path"])
print("Embedding source:", result.get("embedding_source"))
"""

    return notebook_object(
        [
            markdown_cell(md),
            code_cell(install_code),
            code_cell(setup_code),
            code_cell(core_source),
            code_cell(run_code),
            code_cell(inspect_code),
            code_cell(save_code),
        ]
    )


def make_comparison_notebook(core_source: str) -> dict:
    method_ids = [spec.method_id for spec in METHOD_SPECS]

    md = """
# Indexing Method Comparison (LlamaIndex)

This notebook aggregates method outputs from `results/` and provides ranking + category summaries.
"""

    install_code = """
%pip install -q pandas
"""

    setup_code = """
from pathlib import Path
import json
import pandas as pd

repo_root = Path.cwd()
if not (repo_root / "results").exists():
    repo_root = repo_root.parent

results_dir = repo_root / "results"
"""

    list_code = f"""
EXPECTED_METHOD_IDS = {method_ids!r}

rows = []
for path in sorted(results_dir.glob("*.json")):
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics", {{}})
    rows.append(
        {{
            "method_id": payload.get("method_id"),
            "method_name": payload.get("method_name"),
            "category": payload.get("category"),
            "embedding_source": payload.get("embedding_source"),
            "precision_at_k": metrics.get("precision_at_k", 0.0),
            "recall_at_k": metrics.get("recall_at_k", 0.0),
            "mrr": metrics.get("mrr", 0.0),
            "ndcg_at_k": metrics.get("ndcg_at_k", 0.0),
            "nodes": payload.get("nodes", 0),
            "result_file": path.name,
        }}
    )

score_df = pd.DataFrame(rows)
score_df.sort_values(["mrr", "ndcg_at_k"], ascending=False).reset_index(drop=True)
"""

    summary_code = """
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
available = set(score_df["method_id"].tolist()) if len(score_df) else set()
missing = sorted(set(EXPECTED_METHOD_IDS) - available)

print(f"Expected methods: {len(EXPECTED_METHOD_IDS)}")
print(f"Available result files: {len(available)}")
print("Missing methods:")
missing
"""

    return notebook_object(
        [
            markdown_cell(md),
            code_cell(install_code),
            code_cell(setup_code),
            code_cell(list_code),
            code_cell(summary_code),
            code_cell(coverage_code),
        ]
    )


def write_notebook(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    out_dir = repo_root / "indexing_notebooks"

    core_source = load_core_source(repo_root)

    generated = []
    for spec in METHOD_SPECS:
        filename = f"{method_slug(spec.method_id)}_{spec.name.lower().replace(' ', '_')}.ipynb"
        filename = filename.replace("-", "_")
        nb_path = out_dir / filename
        nb = make_method_notebook(spec.method_id, spec.category, spec.name, spec.description, core_source)
        write_notebook(nb_path, nb)
        generated.append(nb_path)

    compare_path = out_dir / "compare_indexing_methods.ipynb"
    write_notebook(compare_path, make_comparison_notebook(core_source))

    print(f"Generated method notebooks: {len(generated)}")
    print(f"Generated comparison notebook: {compare_path}")


if __name__ == "__main__":
    main()

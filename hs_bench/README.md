# The Hybrid Search Evaluation Framework

---

## 📖 Overview

This `hs_bench/` directory is designed for extending our work. You should use this directory if you want to:

- Add a new dataset to the framework.
- Evaluate a new embedding model (dense, sparse, or tensor).

## 🏛️ Framework Structure

This framework is organized by function. All operations are driven by configuration files (`configs/`) and a set of scripts that correspond to different stages of the evaluation pipeline.

-   **`configs/`**: Contains `.json` config files for each dataset, defining file paths, model names, and index parameters.
-   **`embedding/`**: Scripts for generating embeddings from raw text (`run_embedding.py`).
-   **`import/`**: Script for creating tables and importing data (`import_data.py`).
-   **`indexing/`**: Script for building indexes (`build_index.py`) on existing data.
-   **`retrieval/`**: Scripts for the retrieval process, which includes executing search queries (`run_search.py`) and optionally applying external rerankers (`apply_reranker.py`).
-   **`evaluation/`**: Scripts for parsing logs and evaluating metrics (`evaluate.py`).

## 🚀 End-to-End Workflow

Here is the step-by-step process for running an experiment.

### Step 1: Generate Embeddings
The `embedding/run_embedding.py` script generates embeddings from the raw text in the corpus.

-   **Example**:
    ```bash
    python hs_bench/embedding/run_embedding.py --config mldr_en --model_name bge-m3
    ```

### Step 2: Create Table and Import Data
The `import/import_data.py` script creates a table in Infinity and imports the corpus data.

-   **Example**:
    ```bash
    python hs_bench/import/import_data.py --config mldr_en
    ```

### Step 3: Build Index
The `indexing/build_index.py` script builds a specified index (e.g., `fts`, `dvs`) on the imported data.

-   **Example**:
    ```bash
    # Build a Full-Text Search (FTS) index
    python hs_bench/indexing/build_index.py --config mldr_en --index_type fts

    # Build a Dense Vector Search (DVS) index
    python hs_bench/indexing/build_index.py --config mldr_en --index_type dvs --model_name bge-m3
    ```

### Step 4: Run Retrieval (Search and Rerank)
Run hybrid search queries with `retrieval/run_search.py` and optionally apply a reranker with `reranking/apply_reranker.py`.

-   **Example (Search)**:
    ```bash
    python hs_bench/retrieval/run_search.py --config mldr_en --paths "fts,dvs" --fusion "rrf" --k0 100
    ```
-   **Example (Rerank)**:
    ```bash
    python hs_bench/reranking/apply_reranker.py --reranker_name bge --input_file hs_bench/results/mldr_en_fts_dvs_rrf_k100.trec --config mldr_en
    ```

### Step 5: Evaluate Results
The `evaluation/evaluate.py` script evaluates search results and creates a single CSV summary of all performance and effectiveness metrics. You can optionally filter by configuration.

-   **Example**:
    ```bash
    # Evaluate all runs for the mldr_en config
    python hs_bench/evaluation/evaluate.py --config mldr_en

    # Evaluate all runs and specify custom metrics and output file
    python hs_bench/evaluation/evaluate.py --metrics "ndcg@10" "recall@10" "recall@100" --output_file "hs_bench/results/my_summary.csv"
    ```

## 💡 Extension Guide

### A. How to Add a New Dataset (e.g., "my_dataset")

1.  **Prepare Data**: Place your dataset files (e.g., `corpus.jsonl`, `queries.jsonl`, `qrels/test.tsv`) in a new directory: `hs_bench/datasets/my_dataset/`.
2.  **Create Config**: Create `hs_bench/configs/my_dataset.json`. You can copy an existing config as a template and update the `paths` section to point to your new files.
3.  **Run Pipeline**: You can now run the full 5-step workflow using your new dataset by referencing its config name (e.g., `--config my_dataset`).

### B. How to Add a New Embedding Model (e.g., "my_model")

1.  **Edit Config File**: Open the relevant config file in `hs_bench/configs/` (e.g., `mldr_en.json`).
2.  **Add Model Definition**: Add a new JSON object to the `embedding_models` array. You must specify the `name`, `type`, `dim`, and any model-specific `params`. For a standard HuggingFace model, you would add:
    ```json
    {
      "name": "my_model",
      "type": "huggingface",
      "dim": 768,
      "params": {
        "model_name": "huggingface_user/my_model_name"
      }
    }
    ```
3.  **Run Pipeline**: You can now use your model by specifying `--model_name my_model` when running the workflow.
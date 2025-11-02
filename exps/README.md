# Reproducing Paper Experiments

## 📖 Overview

This directory provides all the necessary scripts and configurations to reproduce the experimental results reported in the paper.

The workflow is designed to be driven by configuration files, allowing for clear and repeatable execution of the experiments for each dataset.

## 🏛️ Directory Structure

-   **`configs/`**: Contains `.json` configuration files for each dataset used in the paper. These files define dataset paths, model parameters, and indexing settings.
-   **`scripts/`**: Contains the core Python scripts for running the different stages of the pipeline:
    -   `run_data_prep.py`: Script to load data and build indexes.
    -   `run_search.py`: Script to execute search queries.
    -   `run_evaluation.py`: Script to evaluate the search results.
-   **`datasets/`**: Contains instructions on how to download the required datasets. See `datasets/README.md`.
-   **`results/`**: The default output directory for search results (`.trec` files) and evaluation summaries (`.csv` files).

## 🚀 How to Run Experiments

### Step 1: Download Datasets

Please follow the instructions in `datasets/README.md` to download and place the datasets in the correct location.

### Step 2: Generate Embeddings

Before importing the data, you need to generate the dense, sparse, and/or tensor embeddings for the corpus and queries. Use the `run_embedding.py` script for this.

This script iterates through the `embedding_models` list in the specified configuration file (e.g., `hs_bench/configs/mldr_en.json`) and generates embeddings for each model. You can control which model to use via the `--model_name` parameter.

**Example:**
To generate embeddings for a specific model:
```bash
python hs_bench/embedding/run_embedding.py --config hs_bench/configs/mldr_en.json --model_name bge-m3
```

To generate embeddings for all models defined in the config, simply omit the `--model_name` argument.

### Step 3: Load Data and Build Indexes

Use the `run_data_prep.py` script with a specific configuration file to create the necessary tables in Infinity and build the required indexes.

**Example:**
```bash
python exps/scripts/run_data_prep.py --config exps/configs/mldr_en.json
```

### Step 4: Run Search Queries

Use the `run_search.py` script to perform the search experiments defined in the configuration file.

**Example:**
```bash
python exps/scripts/run_search.py --config exps/configs/mldr_en.json
```

### Step 5: Evaluate Results

Use the `run_evaluation.py` script to calculate the metrics (e.g., nDCG@10, Recall@100) for the generated search results.

**Example:**
```bash
python exps/scripts/run_evaluation.py --config exps/configs/mldr_en.json
```
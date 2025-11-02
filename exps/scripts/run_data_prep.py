import argparse
import json
import struct
from pathlib import Path

import infinity
import numpy as np
import pandas as pd
from infinity.common import ConflictType
from infinity.index import IndexInfo


def load_dense_vectors(path: Path):
    """Loads dense vectors from a .npy file."""
    if not path.exists():
        raise FileNotFoundError(f"Dense vector file not found: {path}")
    print(f"Loading dense vectors from {path}...")
    return np.load(path)


def load_sparse_vectors(path: Path):
    """Loads sparse vectors from a .json file."""
    if not path.exists():
        raise FileNotFoundError(f"Sparse vector file not found: {path}")
    print(f"Loading sparse vectors from {path}...")
    with open(path, 'r') as f:
        return json.load(f)


def load_tensor_vectors(path: Path):
    """Loads tensor (ColBERT) vectors from a custom .data file."""
    if not path.exists():
        raise FileNotFoundError(f"Tensor vector file not found: {path}")
    print(f"Loading tensor vectors from {path}...")

    all_embeddings = []
    with open(path, 'rb') as f:
        while True:
            len_bytes = f.read(4)
            if not len_bytes:
                break
            length = struct.unpack('<i', len_bytes)[0]

            doc_embeddings = []
            for _ in range(length):
                dim_bytes = f.read(4)
                if not dim_bytes:
                    raise IOError("Incomplete data: failed to read dimension.")
                dim = struct.unpack('<i', dim_bytes)[0]

                vec_bytes = f.read(dim * 4)
                if len(vec_bytes) != dim * 4:
                    raise IOError("Incomplete data: failed to read vector data.")

                vec = np.frombuffer(vec_bytes, dtype=np.float32)
                doc_embeddings.append(vec)
            all_embeddings.append(np.array(doc_embeddings, dtype=np.float32))
    return all_embeddings


def resolve_path(project_root: Path, config_path: str) -> Path:
    """Resolves a path from the config, making it absolute if it's not already."""
    path = Path(config_path)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def run_data_prep(config_path: str):
    """
    Main function to prepare data: connect to DB, create table,
    load data from multiple sources, insert it, and create indexes.
    """
    project_root = Path(__file__).parent.parent.parent.resolve()
    config_path_abs = resolve_path(project_root, config_path)

    print(f"Project Root: {project_root}")
    print(f"Loading config from: {config_path_abs}")

    with open(config_path_abs, 'r') as f:
        config = json.load(f)

    # --- Connect to Infinity and set up table ---
    try:
        infinity_instance = infinity.connect(infinity.common.LOCAL_HOST)
        db = infinity_instance.get_database(config['db_name'])
        db.drop_table(config['table_name'], conflict_type=ConflictType.Ignore)
        table = db.create_table(config['table_name'], config['schema'])
        print(f"Database '{config['db_name']}' and table '{config['table_name']}' are set up.")
    except Exception as e:
        print(f"Error setting up database/table: {e}")
        return

    # --- Load all data sources ---
    try:
        corpus_path = resolve_path(project_root, config['data']['corpus_path'])
        print(f"Loading corpus from {corpus_path}...")
        corpus_df = pd.read_csv(corpus_path)
        num_docs = len(corpus_df)
        print(f"Loaded {num_docs} documents from corpus.")

        records = corpus_df.to_dict('records')

        schema_keys = list(config['schema'].keys())
        
        # Note: We assume the config's `..._dir` keys point to single files, not directories.
        if "dense_vec" in schema_keys and 'dense_embeddings_dir' in config['data']:
            dense_path = resolve_path(project_root, config['data']['dense_embeddings_dir'])
            dense_vectors = load_dense_vectors(dense_path)
            assert num_docs == len(dense_vectors), "Mismatch between corpus and dense vectors count."
            for i in range(num_docs):
                records[i]['dense_vec'] = dense_vectors[i]

        if "sparse_vec" in schema_keys and 'sparse_embeddings_dir' in config['data']:
            sparse_path = resolve_path(project_root, config['data']['sparse_embeddings_dir'])
            sparse_vectors = load_sparse_vectors(sparse_path)
            assert num_docs == len(sparse_vectors), "Mismatch between corpus and sparse vectors count."
            for i in range(num_docs):
                records[i]['sparse_vec'] = sparse_vectors[i]

        if "tensor_vec" in schema_keys and 'tensor_embeddings_dir' in config['data']:
            tensor_path = resolve_path(project_root, config['data']['tensor_embeddings_dir'])
            tensor_vectors = load_tensor_vectors(tensor_path)
            assert num_docs == len(tensor_vectors), "Mismatch between corpus and tensor vectors count."
            for i in range(num_docs):
                records[i]['tensor_vec'] = tensor_vectors[i]

    except (FileNotFoundError, AssertionError, KeyError) as e:
        print(f"Error loading data: {e}")
        return

    # --- Insert data in batches ---
    batch_size = 1000
    print(f"Inserting {num_docs} records in batches of {batch_size}...")
    for i in range(0, num_docs, batch_size):
        batch = records[i:i + batch_size]
        try:
            table.insert(batch)
            print(f"  - Inserted batch {i // batch_size + 1}/{(num_docs + batch_size - 1) // batch_size}")
        except Exception as e:
            print(f"Error inserting batch {i}: {e}")
            return
    print("Data insertion finished.")

    # --- Create indexes ---
    print("Creating indexes...")
    for idx in config.get('indexes', []):
        try:
            print(f"  - Creating index: {idx['name']} on column {idx['column']}")
            index_info = IndexInfo(idx['column'], idx['type'], idx.get('params', {}))
            table.create_index(idx['name'], [index_info], conflict_type=ConflictType.Ignore)
            if 'optimize' in idx:
                print(f"  - Optimizing index: {idx['name']}")
                table.optimize(idx['name'])
        except Exception as e:
            print(f"Error creating index {idx['name']}: {e}")
    print("Index creation finished.")
    print("Data preparation script finished successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import data and build indexes based on a JSON config.")
    parser.add_argument("--config", type=str, required=True, help="Path to the JSON configuration file.")
    args = parser.parse_args()
    run_data_prep(args.config)
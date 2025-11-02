import argparse
import json
from pathlib import Path
import sys
import numpy as np

# Make the project root directory available
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root / 'python'))

import infinity
from infinity.common import LOCAL_HOST, ConflictType, SparseVector

def get_args():
    parser = argparse.ArgumentParser(description="Import data into Infinity database")
    parser.add_argument("--config", type=str, required=True, help="Dataset config name (e.g., mldr_en)")
    return parser.parse_args()

def main():
    args = get_args()
    config_path = project_root / "hs_bench" / "configs" / f"{args.config}.json"

    print(f"Loading config from: {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)

    db_name = config["database"]["name"]
    table_name = config["database"]["table_name"]
    
    # --- Connect to DB ---
    print("Connecting to Infinity database...")
    infinity_obj = infinity.connect(LOCAL_HOST)
    db = infinity_obj.get_database(db_name)
    db.drop_table(table_name, if_exists=True)

    # --- Define Schema --- dynamically based on config
    schema_builder = {
        "_id": {"type": "varchar"},
        "title": {"type": "varchar"},
        "text": {"type": "varchar"},
    }
    for model in config["embedding_models"]:
        col_name = f"embedding_{model['name'].replace('-', '_')}"
        if model['storage_type'] == 'dense':
            schema_builder[col_name] = {"type": f"vector,{model['dim']},float"}
        elif model['storage_type'] == 'sparse':
            schema_builder[col_name] = {"type": f"sparse,{model['dim']},float,int"}
        elif model['storage_type'] == 'tensor':
            schema_builder[col_name] = {"type": f"tensor,{model['dim']},float"}


    print(f"Creating table '{table_name}' in database '{db_name}' with schema: {schema_builder}")
    table = db.create_table(table_name, schema_builder, ConflictType.Error)
    
    # --- Load Corpus ---
    corpus_path = project_root / config["dataset"]["corpus_path"]
    print(f"Loading base corpus from: {corpus_path}")
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")
    
    corpus_data = []
    with open(corpus_path, 'r') as f:
        for line in f:
            corpus_data.append(json.loads(line))

    # --- Load and Prepare Embeddings (Memory-mapped) ---
    embedding_data = {}
    for model in config["embedding_models"]:
        model_name = model['name']
        col_name = f"embedding_{model_name.replace('-', '_')}"
        embedding_file = project_root / config["dataset"]["corpus_embeddings_dir"] / f"{model_name}.npy"
        print(f"Memory-mapping embeddings for '{model_name}' from {embedding_file}")
        if not embedding_file.exists():
            # For this example, if an embedding file is missing, we'll create a dummy one.
            # In a real scenario, you should ensure all files are present.
            print(f"Warning: Embedding file not found: {embedding_file}. Creating a dummy file.")
            dummy_shape = (len(corpus_data), model['dim']) if model['storage_type'] in ['vector', 'tensor'] else (len(corpus_data),)
            if model['storage_type'] == 'sparse':
                dummy_data = np.array([{} for _ in range(len(corpus_data))], dtype=object)
            else:
                dummy_data = np.zeros(dummy_shape, dtype=np.float32)
            np.save(embedding_file, dummy_data)

        embedding_data[col_name] = np.load(embedding_file, allow_pickle=True, mmap_mode='r')

    # --- Batch Insert Data ---
    print(f"Importing data into table '{table_name}'...")
    batch_size = 500
    buffer = []
    total_docs = len(corpus_data)
    for i in range(total_docs):
        doc = corpus_data[i]
        insert_record = {
            "_id": doc["_id"],
            "title": doc.get("title", ""),
            "text": doc.get("text", ""),
        }
        for model in config["embedding_models"]:
            col_name = f"embedding_{model['name'].replace('-', '_')}"
            embedding = embedding_data[col_name][i]
            if model['storage_type'] == 'sparse':
                indices = [int(k) for k in embedding.keys()]
                values = list(embedding.values())
                insert_record[col_name] = SparseVector(indices, values)
            else: # Handles both 'vector' and 'tensor'
                insert_record[col_name] = embedding.tolist()
        
        buffer.append(insert_record)
        if len(buffer) >= batch_size:
            table.insert(buffer)
            buffer = []
            print(f"Inserted {i + 1}/{total_docs} records...")

    if buffer:
        table.insert(buffer)
    
    print(f"Data import completed successfully. Inserted {total_docs} records.")

if __name__ == "__main__":
    main()
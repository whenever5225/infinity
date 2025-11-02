import argparse
import json
from pathlib import Path
import sys

# Make the project root directory available
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root / 'python'))

import infinity
from infinity.common import LOCAL_HOST
import infinity.index as index

def get_args():
    parser = argparse.ArgumentParser(description="Build an index in Infinity")
    parser.add_argument("--config", type=str, required=True, help="Dataset config name")
    parser.add_argument("--index_type", type=str, required=True, choices=["fts", "dvs", "hnsw", "bmp", "emvb"], help="Type of index to build")
    parser.add_argument("--model_name", type=str, help="Model name, required for vector/sparse/tensor indexes to identify the column")
    return parser.parse_args()

def main():
    args = get_args()
    if args.index_type == "dvs":
        args.index_type = "hnsw" # Treat dvs as an alias for hnsw

    if args.index_type != "fts" and not args.model_name:
        raise ValueError(f"Argument --model_name is required for building a '{args.index_type}' index.")

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
    table = db.get_table(table_name)

    # --- Build Index ---
    if args.index_type == "fts":
        index_name = f"idx_fts_{table_name}"
        print(f"Building FTS index '{index_name}' on 'text' and 'title' columns...")
        table.drop_index(index_name, if_exists=True)
        table.create_index(index_name, [
            index.IndexInfo("title", index.IndexType.FullText, []),
            index.IndexInfo("text", index.IndexType.FullText, []),
        ])
        print("FTS index built successfully.")

    else: # Handle vector, sparse, and tensor indexes
        col_name = f"embedding_{args.model_name.replace('-', '_')}"
        index_name = f"idx_{args.index_type}_{col_name}"
        
        # Get index params from config
        index_params_config = config.get("index_params", {}).get(args.index_type, {})
        if not index_params_config:
            raise ValueError(f"No index parameters found for type '{args.index_type}' in config.")

        params = [index.InitParameter(k, str(v)) for k, v in index_params_config.items()]
        
        index_type_map = {
            "hnsw": index.IndexType.Hnsw,
            "bmp": index.IndexType.BMP,
            "emvb": index.IndexType.EMVB,
        }
        infinity_index_type = index_type_map[args.index_type]

        print(f"Building {args.index_type.upper()} index '{index_name}' on column '{col_name}'...")
        print(f"With parameters: {index_params_config}")
        
        table.drop_index(index_name, if_exists=True)
        table.create_index(index_name, [
            index.IndexInfo(col_name, infinity_index_type, params)
        ])
        print(f"{args.index_type.upper()} index built successfully.")

if __name__ == "__main__":
    main()
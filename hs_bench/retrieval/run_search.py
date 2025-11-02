import argparse
import json
from pathlib import Path
import sys
import numpy as np
from scipy.sparse import load_npz

# Make the project root directory available
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root / 'python'))

import infinity
from infinity.common import LOCAL_HOST, SparseVector

def get_args():
    parser = argparse.ArgumentParser(description="Run a hybrid search query in Infinity")
    parser.add_argument("--config", type=str, required=True, help="Dataset config name")
    parser.add_argument("--paths", nargs='+', required=True, help="Paths to the models/methods to use for search (e.g., fts, bge-large-en-v1.5)")
    parser.add_argument("--fusion", type=str, default="rrf", choices=["rrf"], help="Fusion method")
    parser.add_argument("--k", type=int, default=100, help="Number of results to retrieve for each search path")
    return parser.parse_args()

def load_query_embeddings(query_embed_dir, model_name):
    # for dense and tensor
    path = query_embed_dir / f"{model_name}.npy"
    if path.exists():
        print(f"Loading dense/tensor query embeddings from {path}")
        return np.load(path)
    
    # for sparse
    path = query_embed_dir / f"{model_name}.npz"
    if path.exists():
        print(f"Loading sparse query embeddings from {path}")
        sparse_matrix = load_npz(path)
        return [SparseVector(sparse_matrix.indices[sparse_matrix.indptr[i]:sparse_matrix.indptr[i+1]].tolist(), 
                             sparse_matrix.data[sparse_matrix.indptr[i]:sparse_matrix.indptr[i+1]].tolist())
                for i in range(sparse_matrix.shape[0])]
    
    raise FileNotFoundError(f"Could not find query embeddings for model {model_name} in {query_embed_dir}")

def main():
    args = get_args()
    config_path = project_root / "hs_bench" / "configs" / f"{args.config}.json"
    print(f"Loading config from: {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)

    db_name = config["database"]["name"]
    table_name = config["database"]["table_name"]
    queries_path = project_root / config["dataset"]["queries_path"]
    query_embed_dir = project_root / config["dataset"]["queries_embeddings_dir"]
    output_dir = project_root / "hs_bench" / "results"
    output_dir.mkdir(exist_ok=True)

    # --- Connect to DB ---
    print("Connecting to Infinity database...")
    infinity_obj = infinity.connect(LOCAL_HOST)
    db = infinity_obj.get_database(db_name)
    table = db.get_table(table_name)

    # --- Load Queries ---
    print(f"Loading queries from {queries_path}...")
    queries = []
    with open(queries_path, 'r') as f:
        for line in f:
            qid, qtext = line.strip().split('\t')
            queries.append((qid, qtext))
            
    # --- Resolve search paths (e.g., map "dvs" to a concrete model name) ---
    resolved_paths = []
    for path in args.paths:
        if path == "dvs":
            # Find the dense model in the config
            dense_model = next((m for m in config["embedding_models"] if m.get("storage_type") == "dense"), None)
            if not dense_model:
                raise ValueError("'dvs' path specified, but no model with storage_type 'dense' found in config.")
            resolved_paths.append(dense_model["name"])
        else:
            resolved_paths.append(path)
    print(f"Resolved search paths: {resolved_paths}")

    # --- Load all necessary query embeddings ---
    query_embeddings = {}
    for path in resolved_paths:
        if path != "fts":
            query_embeddings[path] = load_query_embeddings(query_embed_dir, path)

    # --- Run Search ---
    run_name = f"{args.config}_{'_'.join(args.paths)}_{args.fusion}_k{args.k}"
    output_path = output_dir / f"{run_name}.trec"
    print(f"Running search and saving results to {output_path}")

    with open(output_path, 'w') as f:
        for i, (qid, qtext) in enumerate(queries):
            # Build the chained query
            search_query = table.output(["_row_id", "_score"])
            
            fusion_queries = []
            for path in resolved_paths:
                model_config = next((m for m in config["embedding_models"] if m["name"] == path), None)
                
                if path == "fts":
                    fusion_queries.append(f"match(title, '{qtext}', 'topn={args.k}')")
                elif model_config:
                    storage_type = model_config.get("storage_type", "dense") # Default to dense
                    embedding_col = f'embedding_{model_config["name"].replace("-", "_")}'
                    query_emb = query_embeddings[path][i]

                    if storage_type == "dense":
                        fusion_queries.append(f"knn({embedding_col}, {query_emb.tolist()}, 'float', 'ip', {args.k})")
                    elif storage_type == "sparse":
                        fusion_queries.append(f"match_sparse({embedding_col}, {query_emb}, 'ip', {args.k})")
                    elif storage_type == "tensor":
                         fusion_queries.append(f"match_tensor({embedding_col}, {query_emb.tolist()}, 'float', 'ip', {args.k})")

            # Apply fusion
            if len(fusion_queries) > 1:
                search_query = search_query.fusion(method=args.fusion, topn=args.k, query_list=fusion_queries)
            else: # Handle single path case
                if resolved_paths[0] == "fts":
                     search_query = search_query.match([], f"title:'{qtext}' OR text:'{qtext}'", f"topn={args.k}")
                else:
                    model_config = next((m for m in config["embedding_models"] if m["name"] == resolved_paths[0]), None)
                    storage_type = model_config.get("storage_type", "dense")
                    embedding_col = f'embedding_{model_config["name"].replace("-", "_")}'
                    query_emb = query_embeddings[resolved_paths[0]][i]
                    if storage_type == "dense":
                        search_query = search_query.knn(embedding_col, query_emb.tolist(), "ip", args.k)
                    elif storage_type == "sparse":
                        search_query = search_query.match_sparse(embedding_col, query_emb, "ip", args.k)
                    elif storage_type == "tensor":
                        search_query = search_query.match_tensor(embedding_col, query_emb.tolist(), "ip", args.k)

            # Execute and write results
            res_df = search_query.to_df()
            for rank, row in res_df.iterrows():
                f.write(f"{qid} Q0 {row['_row_id']} {rank + 1} {row['_score']:.6f} {run_name}\n")

    print("Search complete.")

if __name__ == "__main__":
    main()
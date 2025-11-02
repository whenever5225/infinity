import argparse
import json
import struct
from collections import defaultdict
from pathlib import Path

import infinity
import numpy as np
from infinity.errors import ErrorCode


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


def reciprocal_rank_fusion(results_map: dict, k: int = 60, limit: int = 100):
    """
    Performs Reciprocal Rank Fusion on a map of search results.
    """
    fused_scores = defaultdict(float)

    for search_type, results in results_map.items():
        for rank, doc in enumerate(results):
            doc_id = doc['docid']
            fused_scores[doc_id] += 1 / (k + rank)

    reranked_doc_ids = sorted(fused_scores.keys(), key=lambda id: fused_scores[id], reverse=True)

    fused_results = []
    for doc_id in reranked_doc_ids[:limit]:
        fused_result = {
            'docid': doc_id,
            '_score': fused_scores[doc_id]
        }
        fused_results.append(fused_result)

    return fused_results


def run_search(config: dict, project_root: Path):
    """
    Executes search tasks defined in the config file.
    """
    try:
        infinity_instance = infinity.connect(infinity.common.LOCAL_HOST)
        db = infinity_instance.get_database(config['db_name'])
        table = db.get_table(config['table_name'])
    except Exception as e:
        print(f"Error connecting to database/table: {e}")
        return

    for task in config.get('search_tasks', []):
        print(f"--- Running search task: {task['name']} ---")

        queries = {}
        if 'query_file' in task:
            queries['dense'] = load_dense_vectors(resolve_path(project_root, task['query_file']))
        if 'query_sparse_file' in task:
            queries['sparse'] = load_sparse_vectors(resolve_path(project_root, task['query_sparse_file']))
        if 'query_tensor_file' in task:
            queries['tensor'] = load_tensor_vectors(resolve_path(project_root, task['query_tensor_file']))
        if 'query_fulltext_file' in task:
            with open(resolve_path(project_root, task['query_fulltext_file']), 'r') as f:
                queries['fulltext'] = [line.strip() for line in f.readlines()]

        if not queries:
            print(f"Warning: No query files found for task '{task['name']}'. Skipping.")
            continue

        num_queries = len(next(iter(queries.values())))
        print(f"Loaded {num_queries} queries.")

        output_path = resolve_path(project_root, task['output_file'])
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f_out:
            for i in range(num_queries):
                query_id = i + 1

                search_type = task['search_type']
                top_k = task.get('top_k', 100)

                if search_type == 'hybrid':
                    all_results = {}
                    for sub_search in task.get('searches', []):
                        builder = table.query_builder()
                        builder.output(['docid', '_score'])

                        search_method = sub_search['type']
                        query_data = queries[search_method][i]

                        if search_method == 'dense':
                            builder.knn('dense_vec', query_data.tolist(), 'float32', sub_search['metric'], top_k)
                        elif search_method == 'sparse':
                            builder.match_sparse('sparse_vec', query_data, 'bm25', top_k)
                        elif search_method == 'tensor':
                            builder.match_tensor('tensor_vec', query_data.tolist(), 'float32', 'maxsim', top_k)
                        elif search_method == 'fulltext':
                            builder.match('text', query_data, f'topn={top_k}')

                        res = builder.search()
                        if res.error_code == ErrorCode.OK:
                            all_results[search_method] = res.to_dicts()
                        else:
                            print(f"Error during {search_method} search for query {query_id}: {res.error_msg}")

                    fusion_method = task.get('fusion', {}).get('method', 'rrf')
                    if fusion_method == 'rrf':
                        final_results = reciprocal_rank_fusion(all_results, k=task['fusion'].get('k', 60), limit=top_k)
                    else:
                        print(f"Warning: Unknown fusion method '{fusion_method}'. Defaulting to first result set.")
                        final_results = next(iter(all_results.values()), [])

                else:
                    builder = table.query_builder()
                    builder.output(['docid', '_score'])
                    query_data = queries[search_type][i]

                    if search_type == 'dense':
                        builder.knn('dense_vec', query_data.tolist(), 'float32', task['metric'], top_k)
                    elif search_type == 'sparse':
                        builder.match_sparse('sparse_vec', query_data, 'bm25', top_k)
                    elif search_type == 'tensor':
                        builder.match_tensor('tensor_vec', query_data.tolist(), 'float32', 'maxsim', top_k)
                    elif search_type == 'fulltext':
                        builder.match('text', query_data, f'topn={top_k}')

                    res = builder.search()
                    if res.error_code == ErrorCode.OK:
                        final_results = res.to_dicts()
                    else:
                        print(f"Error during {search_type} search for query {query_id}: {res.error_msg}")
                        final_results = []

                for rank, hit in enumerate(final_results):
                    f_out.write(f"{query_id} Q0 {hit['docid']} {rank + 1} {hit['_score']} {task['name']}\n")

                if (i + 1) % 10 == 0:
                    print(f"  - Processed {i + 1}/{num_queries} queries.")

        print(f"--- Task '{task['name']}' finished. Results saved to {output_path} ---")


def main():
    parser = argparse.ArgumentParser(description="Run search experiments based on a JSON config.")
    parser.add_argument("--config", type=str, required=True, help="Path to the JSON configuration file.")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent.parent.resolve()
    config_path_abs = resolve_path(project_root, args.config)

    with open(config_path_abs, 'r') as f:
        config = json.load(f)

    run_search(config, project_root)


if __name__ == "__main__":
    main()
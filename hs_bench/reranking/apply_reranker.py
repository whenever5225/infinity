import argparse
import json
from pathlib import Path
import sys
from collections import defaultdict
from FlagEmbedding import FlagReranker
from tqdm import tqdm

# Make the project root directory available
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root / 'python'))

def get_args():
    parser = argparse.ArgumentParser(description="Apply a reranker to a TREC run file")
    parser.add_argument("--reranker_name", type=str, required=True, help="Name or path of the reranker model (e.g., 'BAAI/bge-reranker-v2-m3')")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input TREC run file")
    parser.add_argument("--config", type=str, required=True, help="Dataset config name")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for reranker model")
    return parser.parse_args()

def main():
    args = get_args()
    input_path = Path(args.input_file)
    if not input_path.is_absolute():
        input_path = project_root / input_path

    # --- Load Config ---
    config_path = project_root / "hs_bench" / "configs" / f"{args.config}.json"
    print(f"Loading config from: {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)

    # --- Load Corpus and Queries ---
    corpus_path = project_root / config["dataset"]["corpus_path"]
    queries_path = project_root / config["dataset"]["queries_path"]

    print(f"Loading corpus from {corpus_path}...")
    corpus = {}
    with open(corpus_path, 'r') as f:
        for line in f:
            doc = json.loads(line)
            corpus[doc['_id']] = doc.get("title", "") + " " + doc.get("text", "")

    print(f"Loading queries from {queries_path}...")
    queries = {}
    with open(queries_path, 'r') as f:
        for line in f:
            qid, qtext = line.strip().split('\t')
            queries[qid] = qtext

    # --- Load Reranker Model ---
    print(f"Loading reranker model: {args.reranker_name}")
    reranker = FlagReranker(args.reranker_name, use_fp16=True)

    # --- Group run file by query ID ---
    print(f"Reading and grouping run file: {input_path}")
    run = defaultdict(list)
    with open(input_path, 'r') as f:
        for line in f:
            qid, _, docid, _, initial_score, run_name = line.strip().split()
            run[qid].append((docid, float(initial_score)))

    # --- Rerank and Write Output ---
    reranker_id = args.reranker_name.split('/')[-1]
    output_filename = f"{input_path.stem}_reranked_{reranker_id}.trec"
    output_path = input_path.parent / output_filename
    
    print(f"Reranking and writing to {output_path}...")
    with open(output_path, 'w') as fout:
        for qid, doc_scores in tqdm(run.items(), desc="Reranking queries"):
            query_text = queries.get(qid)
            if not query_text:
                print(f"Warning: Query ID {qid} not found in queries file. Skipping.")
                continue

            doc_ids = [ds[0] for ds in doc_scores]
            doc_texts = [corpus.get(did, "") for did in doc_ids]
            
            # Create pairs of [query, document]
            pairs = [[query_text, doc_text] for doc_text in doc_texts]

            # Compute new scores
            new_scores = reranker.compute_score(pairs, batch_size=args.batch_size)

            # Combine doc_ids with new scores and sort
            reranked_results = sorted(zip(doc_ids, new_scores), key=lambda x: x[1], reverse=True)

            # Write to new TREC file
            run_name_reranked = f"{run_name}_reranked_{reranker_id}"
            for rank, (docid, score) in enumerate(reranked_results):
                fout.write(f"{qid} Q0 {docid} {rank + 1} {score:.6f} {run_name_reranked}\n")

    print("Reranking complete.")

if __name__ == "__main__":
    main()
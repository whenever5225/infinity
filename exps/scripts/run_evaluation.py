import argparse
import json
from pathlib import Path
import pandas as pd
import pytrec_eval

def resolve_path(project_root: Path, config_path: str) -> Path:
    """Resolves a path from the config, making it absolute if it's not already."""
    path = Path(config_path)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()

def run_evaluation(config: dict, project_root: Path):
    """Evaluates search results based on the provided configuration."""
    qrels_path = resolve_path(project_root, config['evaluation']['qrels_file'])
    output_summary_path = resolve_path(project_root, config['evaluation']['output_summary_file'])

    print(f"Loading Qrels from: {qrels_path}")
    with open(qrels_path, 'r') as f_qrel:
        qrels = pytrec_eval.parse_qrel(f_qrel)

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'ndcg_cut.10', 'recall.100'})

    all_metrics = []

    for task in config.get('search_tasks', []):
        task_name = task['name']
        result_file_path = resolve_path(project_root, task['output_file'])

        if not result_file_path.exists():
            print(f"Warning: Result file for task '{task_name}' not found at {result_file_path}. Skipping.")
            continue

        print(f"Evaluating task: {task_name} from {result_file_path}")
        with open(result_file_path, 'r') as f_run:
            try:
                run = pytrec_eval.parse_run(f_run)
                results = evaluator.evaluate(run)

                task_metrics = {
                    'task_name': task_name
                }

                # Aggregate metrics across all queries for the task
                aggregated_metrics = defaultdict(float)
                for query_id, query_measures in results.items():
                    for measure, value in query_measures.items():
                        aggregated_metrics[measure] += value
                
                num_queries = len(results)
                if num_queries > 0:
                    for measure, total_value in aggregated_metrics.items():
                        task_metrics[measure] = total_value / num_queries
                
                all_metrics.append(task_metrics)

            except Exception as e:
                print(f"Error evaluating file {result_file_path}: {e}")

    if not all_metrics:
        print("No evaluation results to summarize.")
        return

    # Create DataFrame and save to CSV
    df = pd.DataFrame(all_metrics)
    df = df.round(4) # Round metrics for clarity
    
    # Reorder columns for better readability
    cols = ['task_name', 'ndcg_cut_10', 'recall_100']
    other_cols = [c for c in df.columns if c not in cols]
    df = df[cols + other_cols]

    output_summary_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_summary_path, index=False)

    print(f"\nEvaluation summary saved to: {output_summary_path}")
    print(df.to_string(index=False))

def main():
    parser = argparse.ArgumentParser(description="Evaluate search results using pytrec_eval.")
    parser.add_argument("--config", type=str, required=True, help="Path to the JSON configuration file.")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent.parent.resolve()
    config_path_abs = resolve_path(project_root, args.config)

    with open(config_path_abs, 'r') as f:
        config = json.load(f)
    
    # Need to add collections for defaultdict
    from collections import defaultdict
    run_evaluation(config, project_root)

if __name__ == "__main__":
    main()
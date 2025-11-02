import argparse
import json
from pathlib import Path
import sys
import platform
import subprocess
from pprint import pprint
import csv

# Make the project root directory available
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root / 'python'))

from pyserini.util import download_evaluation_script

def get_args():
    parser = argparse.ArgumentParser(description="Evaluate search runs and create a summary CSV.")
    parser.add_argument("--config", type=str, default=None, help="Optional: Evaluate only runs for a specific config (e.g., mldr_en).")
    parser.add_argument("--metrics", nargs='+', default=["ndcg@10", "recall@100"], help="Metrics to evaluate.")
    parser.add_argument("--output_file", type=str, default="hs_bench/results/evaluation_summary.csv", help="Path to save the summary CSV file.")
    return parser.parse_args()

def map_metric(metric: str):
    metric, k = metric.split('@')
    if metric.lower() == 'ndcg':
        return k, f'ndcg_cut.{k}'
    elif metric.lower() == 'recall':
        return k, f'recall.{k}'
    else:
        raise ValueError(f"Unknown metric: {metric}")

def evaluate(script_path, qrels_path, query_result_path, metrics: list):
    cmd_prefix = ['java', '-jar', script_path]
    results = {}
    for metric in metrics:
        k, mapped_metric = map_metric(metric)
        args = ['-c', '-M', str(k), '-m', mapped_metric, str(qrels_path), str(query_result_path)]
        cmd = cmd_prefix + args

        shell = platform.system() == "Windows"
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=shell)
        stdout, stderr = process.communicate()
        if stderr:
            print(stderr.decode("utf-8"))
        result_str = stdout.decode("utf-8").strip()
        try:
            results[metric] = float(result_str.split('\t')[-1])
        except (ValueError, IndexError):
            print(f"Could not parse result for {metric}: {result_str}")
            results[metric] = result_str
    return results

def main():
    args = get_args()
    results_dir = project_root / "hs_bench" / "results"
    configs_dir = project_root / "hs_bench" / "configs"
    output_path = project_root / args.output_file
    script_path = download_evaluation_script('trec_eval')

    run_files = list(results_dir.glob("*.trec"))

    # Filter by config if provided
    if args.config:
        run_files = [f for f in run_files if f.name.startswith(args.config)]
        print(f"Found {len(run_files)} runs for config '{args.config}'.")

    if not run_files:
        print(f"No .trec files found in {results_dir} matching the criteria. Nothing to evaluate.")
        return

    print(f"Found {len(run_files)} run files to evaluate in total.")
    
    all_run_results = []

    for run_file in run_files:
        try:
            config_name = run_file.name.split('_')[0]
            config_path = configs_dir / f"{config_name}.json"

            if not config_path.exists():
                print(f"Warning: Could not find config file for run '{run_file.name}'. Searched at: {config_path}. Skipping.")
                continue

            with open(config_path, 'r') as f:
                config = json.load(f)
            
            qrels_path = project_root / config["dataset"]["qrels_path"]
            if not qrels_path.exists():
                print(f"Warning: Qrels file not found for config '{config_name}' at {qrels_path}. Skipping evaluation for {run_file.name}.")
                continue

            print("*****************************")
            print(f"Evaluating {run_file.name}...")
            
            result = evaluate(script_path, qrels_path, run_file, args.metrics)
            result['run_name'] = run_file.name
            all_run_results.append(result)
            
            print("--- Evaluation Results ---")
            pprint(result)
            print("============================")

        except Exception as e:
            print(f"An error occurred while processing {run_file.name}: {e}")

    # Write summary CSV
    if not all_run_results:
        print("\nNo results to write to CSV.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ['run_name'] + args.metrics
    
    filtered_results = [{key: res.get(key) for key in fieldnames} for res in all_run_results]

    try:
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(filtered_results)
        print(f"\nEvaluation summary saved to {output_path}")
    except Exception as e:
        print(f"\nError writing to CSV file {output_path}: {e}")

if __name__ == "__main__":
    main()
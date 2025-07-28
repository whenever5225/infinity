from datasets import load_dataset
# JSONL
dataset = load_dataset("json", data_files="/home/ubuntu/data_download_data/data_download/dataset/TREC-COVID_en/trec-covid/queries.jsonl",split="train")
# "text""label"
dataset = dataset.select_columns(["_id", "text"])
# CSVPandas
import pandas as pd
df = dataset.to_pandas()
df.to_csv("/home/ubuntu/hybridsearch/experiments/load_experiments/TREC_COVID_en/queries_with_id.csv", index=False, encoding="utf-8")

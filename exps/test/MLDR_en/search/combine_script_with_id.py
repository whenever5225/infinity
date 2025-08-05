import datasets
from datasets import Dataset 
import pandas as pd

#  CSV 
file_path = '/home/ubuntu/infinity/experiments/load_experiments/MLDR_en/search/mldr_en_query.csv'
df = pd.read_csv(file_path)

# 
columns_to_keep = ['query_id', 'query']

# 
filtered_df = df[columns_to_keep]

#  CSV 
output_file = 'queries_decline_with_id.csv'
filtered_df.to_csv(output_file, index=False)
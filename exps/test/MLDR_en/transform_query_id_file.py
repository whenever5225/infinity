import pandas as pd

# 1. CSV
df = pd.read_csv('/home/ubuntu/data_download_data/embedding_reserve/MLDR_en/mldr_en_query.csv')

# 2. query_idquery
df = df[['query_id', 'query']]

# 3. _idtext
df = df.rename(columns={'query_id': '_id', 'query': 'text'})

# 4. 
df.to_csv("/home/ubuntu/hybridsearch/experiments/load_experiments/MLDR_en/queries_with_id.csv", index=False, encoding="utf-8")

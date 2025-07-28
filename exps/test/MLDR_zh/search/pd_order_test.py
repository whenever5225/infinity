import pandas as pd

#  DataFrame
data = {
    'query-id': [1, 2, 1, 2, 1],
    'value': [10, 20, 30, 40, 50]
}
df = pd.DataFrame(data)

#  'query-id' 
grouped = df.groupby('query-id')
print(grouped)
# 
for query_id, group in grouped:
    print(f"Query ID: {query_id}")
    print(group)
    print("-" * 20)
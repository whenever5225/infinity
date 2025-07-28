

import os


sparse_embedding_dir = '/home/ubuntu/data_download_data/embedding_reserve/mldr_en/query_tensor_embeddings/vectors'
sparse_names = [f for f in os.listdir(sparse_embedding_dir) if os.path.isfile(os.path.join(sparse_embedding_dir, f))]
for sparse_name in sparse_names:
    print(sparse_name)
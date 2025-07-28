import json
import struct
import numpy as np

def load_dense(dense_save_file: str):
    dense_embeddings = []
    with open(dense_save_file, 'rb') as f:
        while True:
            # 4
            dim_data = f.read(4)
            if not dim_data:  # 
                break
            dim = struct.unpack('<i', dim_data)[0]
            
            # 
            vector_data = f.read(dim * 4)  # 4
            one_dense = np.frombuffer(vector_data, dtype=np.float32)
            one_dense = one_dense.astype(np.float16)
            dense_embeddings.append(one_dense)
    return np.array(dense_embeddings)

def load_sparse(sparse_save_file: str):
    with open(sparse_save_file, 'r', encoding='utf-8') as file:
        return json.load(file)

def load_result(dense_save_file: str):
    dense_embeddings = []
    with open(dense_save_file, 'rb') as f:
        while True:
            # 4
            dim_data = f.read(4)
            if not dim_data:  # 
                break
            dim = struct.unpack('<i', dim_data)[0]
            
            # 
            vector_data = f.read(dim * 2)  # 4
            one_dense = np.frombuffer(vector_data, dtype=np.float32)
            
            dense_embeddings.append(one_dense)
    return np.array(dense_embeddings)

# # 
# dense_save_file = "/home/ubuntu/experiments/small_embedding/MLDR_en/dense_embeddings/vectors/MLDR_en_dense1.fvecs"
# dense_embeddings = load_result(dense_save_file)

# print(f"Loaded {len(dense_embeddings)} embeddings with dimension {dense_embeddings[0]}")

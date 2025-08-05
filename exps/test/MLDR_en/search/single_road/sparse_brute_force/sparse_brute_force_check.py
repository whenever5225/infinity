import os
import re
import time
from datasets import Dataset
import pandas as pd
from vec_read import load_sparse
import numpy as np

def GetQuestions():
    sparse_embedding_dir = "/home/ubuntu/data_download_data/embedding_reserve/mldr_en/query_sparse_embeddings/vectors"
    df = pd.read_csv('/home/ubuntu/infinity/experiments/load_experiments/MLDR_en/search/queries_decline_with_id.csv')
    sparse_names = [f for f in os.listdir(sparse_embedding_dir) if os.path.isfile(os.path.join(sparse_embedding_dir, f))]
    sparse_file_idx = 0
    questions = []
    id = 0
    while sparse_file_idx < len(sparse_names):
        sparse_path = os.path.join(sparse_embedding_dir,sparse_names[sparse_file_idx])
        query_sparse_vectors = load_sparse(sparse_path)
        for i in range(len(query_sparse_vectors)):
            indices = []
            values = []
            for key, value in query_sparse_vectors[i].items():
                indices.append(int(key))
                values.append(value)
            questions.append((dict(zip(indices, values)),df.iloc[id]['query_id']))
            id += 1
        sparse_file_idx += 1
    return questions

# 
def extract_number(filename):
    match = re.search(r'sparse(\d+)\.fvecs', filename)
    if match:
        return int(match.group(1))
    return 0

def GetSparseData():
    corpus = Dataset.from_csv('/home/ubuntu/data_download_data/embedding_reserve/mldr_en/mldr_en_corpus.csv')
    total_num = corpus.num_rows
    docid_list = corpus["docid"]
    del corpus
    print(f"Expect total number of rows: {total_num}")
    sparse_embedding_dir = '/home/ubuntu/data_download_data/embedding_reserve/mldr_en/sparse_embedding/vectors'
    sparse_names = [f for f in os.listdir(sparse_embedding_dir) if os.path.isfile(os.path.join(sparse_embedding_dir, f))]
    # 
    sparse_names = sorted(sparse_names, key=extract_number)
    for sparse_name in sparse_names:
        print(sparse_name)
    sparse_file_idx = 0
    doc_id_idx = 0
    sparse_idx = 0
    sparse_nums = 0
    begin_insert_time = time.time()
    sparse_data_vectors = []
    while sparse_file_idx < len(sparse_names):
        sparse_path = os.path.join(sparse_embedding_dir,sparse_names[sparse_file_idx])
        sparse_vectors = load_sparse(sparse_path)
        while sparse_idx < len(sparse_vectors):
            indices = []
            values = []
            sparse_nums += 1
            for key, value in sparse_vectors[sparse_idx].items():
                indices.append(int(key))
                values.append(value)
            sparse_data_vectors.append((dict(zip(indices, values)),docid_list[doc_id_idx]))
            doc_id_idx += 1
            sparse_idx += 1
        if sparse_idx >= len(sparse_vectors):
            sparse_idx = 0
            sparse_file_idx += 1
    end_insert_time = time.time()
    print("insert time: ",(end_insert_time - begin_insert_time)*1000,'ms')
    return sparse_data_vectors

def sparse_vector_inner_product(index_value_dict1, vec2_dict2):
    inner_product = 0
    for idx, val2 in vec2_dict2.items():
        # 
        if idx in index_value_dict1:
            # 
            inner_product += index_value_dict1[idx] * val2
    return inner_product

def search_nearest(query_vectors, dataset, result_file=None,top_k=10):
    """
    top_k
    """
    results = []
    id = 0
    for (query_vector_dict,query_id) in query_vectors:
        similarities = []
        for sparse_vector_dict, doc_id in dataset:
            # 
            similarity = sparse_vector_inner_product(query_vector_dict, sparse_vector_dict)
            similarities.append((query_id, doc_id, similarity))
        
        # 
        similarities.sort(key=lambda x: x[2], reverse=True)
        # top_k
        top_k_results = similarities[:top_k]
        # 
        for result in top_k_results:
            query_id, doc_id, similarity = result
            result_file.write(f"{query_id}\t{doc_id}\n")
            result_file.flush()
        results.append(top_k_results)
        id += 1
        print(id)
    return results

def main():
    questions = GetQuestions()
    sparse_data_vectors = GetSparseData()
    print("questions: ",len(questions))
    print("sparse_data_vectors: ",len(sparse_data_vectors))
    with open('/home/ubuntu/infinity/experiments/load_experiments/MLDR_en/search/single_road/sparse_brute_force' + '/sparse_brute_force_result.txt','w') as result_file:
        result_file.write("query-id\tcorpus-id\n")
        nearest_results = search_nearest(questions, sparse_data_vectors, result_file, top_k=10)

if __name__ == "__main__":
    main()

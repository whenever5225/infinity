import os
import time
from datasets import Dataset
import pandas as pd
from colbert_read import load_colbert_list
import numpy as np

def GetQuestions():
    tensor_embedding_dir = "/home/ubuntu/data_download_data/embedding_reserve/mldr_en/query_tensor_embeddings/vectors"
    df = pd.read_csv('/home/ubuntu/infinity/experiments/load_experiments/MLDR_en/search/queries_decline_with_id.csv')
    tensor_names = [f for f in os.listdir(tensor_embedding_dir) if os.path.isfile(os.path.join(tensor_embedding_dir, f))]
    tensor_file_idx = 0
    questions = []
    id = 0
    while tensor_file_idx < len(tensor_names):
        tensor_path = os.path.join(tensor_embedding_dir,tensor_names[tensor_file_idx])
        query_tensor_vectors = load_colbert_list(tensor_path)
        for i in range(len(query_tensor_vectors)):
            questions.append((query_tensor_vectors[i],df.iloc[id]['query_id']))
            id += 1
        tensor_file_idx += 1
    return questions

def GettensorData():
    corpus = Dataset.from_csv('/home/ubuntu/data_download_data/embedding_reserve/mldr_en/mldr_en_corpus.csv')
    total_num = corpus.num_rows
    docid_list = corpus["docid"]
    del corpus
    print(f"Expect total number of rows: {total_num}")
    tensor_embedding_dir = '/home/ubuntu/data_download_data/embedding_reserve/mldr_en/tensor_embeddings/vectors'
    tensor_names = [f for f in os.listdir(tensor_embedding_dir) if os.path.isfile(os.path.join(tensor_embedding_dir, f))]
    tensor_file_idx = 0
    doc_id_idx = 0
    tensor_idx = 0
    tensor_nums = 0
    begin_insert_time = time.time()
    tensor_data_vectors = []
    while tensor_file_idx < len(tensor_names):
        tensor_path = os.path.join(tensor_embedding_dir,tensor_names[tensor_file_idx])
        tensor_vectors = load_colbert_list(tensor_path)
        while tensor_idx < len(tensor_vectors):
            tensor_nums += 1
            tensor_data_vectors.append((tensor_vectors[tensor_idx],docid_list[doc_id_idx]))
            doc_id_idx += 1
            tensor_idx += 1
            print(doc_id_idx)
        tensor_file_idx += 1
    end_insert_time = time.time()
    print("insert time: ",(end_insert_time - begin_insert_time)*1000,'ms')
    return tensor_data_vectors

def tensor_vector_inner_product(index_value_dict1, vec2_dict2):
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
        for tensor_vector_dict, doc_id in dataset:
            # 
            similarity = tensor_vector_inner_product(query_vector_dict, tensor_vector_dict)
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
    print("questions: ",len(questions))
    tensor_data_vectors = GettensorData()
    print("tensor_data_vectors: ",len(tensor_data_vectors))
    # with open('/home/ubuntu/infinity/experiments/load_experiments/MLDR_en/search/single_road/tensor_brute_force' + '/tensor_brute_force_result.txt','w') as result_file:
    #     result_file.write("query-id\tcorpus-id\n")
    #     nearest_results = search_nearest(questions, tensor_data_vectors, result_file, top_k=10)

if __name__ == "__main__":
    main()

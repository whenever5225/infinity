import os
import numpy as np
import pandas as pd

qrel_file_path = "/home/ubuntu/hybridsearch/experiments/load_experiments/MS_MARCO_en/qrels/test.tsv"
# query_result_file_path = "/home/ubuntu/hybridsearch/experiments/load_experiments/MS_MARCO_en/search1000/four_roads/fulltext_dense_sparse_tensor_result.txt"
path_prefix = "/home/ubuntu/hybridsearch/experiments/load_experiments/MS_MARCO_en/search/"
post_file_paths = [
    ## single_road
    "single_road/sparse_result.txt","single_road/dense_result.txt","single_road/fulltext_result.txt",
    ## two_roads
    "two_roads/dense_sparse_result.txt","two_roads/fulltext_sparse_result.txt","two_roads/fulltext_dense_result.txt",
    ## three_roads 
    "three_roads/fulltext_dense_sparse_result.txt"
]

def load_qrel_map():
    """
     qrel  query-id  corpus-id  score 
    :return: query-id  corpus-id  score 
    """
    #  qrel 
    df = pd.read_csv(qrel_file_path, sep='\t')

    #  query-id 
    grouped = df.groupby('query-id')

    #  query-id  corpus
    result = {}
    for query_id, group in grouped:
        #  corpus-id score 
        mapping = dict(zip(group['corpus-id'], group['score']))
        result[query_id] = mapping
    return result

def calculate_dcg(scores, k=None):
    """
    (DCG)

    :param scores: 
    :param k:  DCG  k  None  DCG
    :return: DCG 
    """
    if k is None:
        k = len(scores)
    if k > len(scores):
        k = len(scores)
    #  k 
    scores = scores[:k]
    #  DCG
    dcg = (2 ** scores[0]) - 1  # 
    for i in range(1, len(scores)):
        dcg += ((2 ** scores[i]) - 1) / np.log2(i + 2)
    return dcg

def cutoff_calc(query_result_file_path,cutoff):
    qrel_map = load_qrel_map()
    df = pd.read_csv(query_result_file_path, sep='\t')
    grouped = df.groupby('query-id')
    result = {}
    for query_id, group in grouped:
        #  corpus-id 
        corpus_ids = group['corpus-id'].tolist()[:cutoff]
        result[query_id] = corpus_ids
    result_ndcg = 0
    delta = 0
    for query_id, corpus_id_list in result.items():
        #  query-id  qrel 
        qrel_mapping = qrel_map.get(query_id, {})
        if qrel_mapping == {}:
            delta += 1
        #  query-id  corpus
        scores_list = [qrel_mapping.get(corpus_id, 0) for corpus_id in corpus_id_list]
        dcg = calculate_dcg(scores_list, 10)
        idcg = calculate_dcg(sorted(scores_list, reverse=True), cutoff)
        ndcg = dcg / idcg if idcg != 0 else 0
        # print(f"Query {query_id} NDCG@10: {ndcg}")
        result_ndcg += ndcg
    return result_ndcg / (len(result)-delta)

def main():
    cutoff = 10
    print(f"cutoff: {cutoff}")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    current_file_path = __file__
    current_file_name = os.path.basename(current_file_path)
    current_file_name_without_extension = os.path.splitext(current_file_name)[0]
    with open(current_dir + "/" + current_file_name_without_extension + ".txt",'w') as tfile:
        for post_file_path in post_file_paths:
            query_result_file_path = path_prefix + post_file_path
            ndcg =  cutoff_calc(query_result_file_path,cutoff)
            print(post_file_path, ndcg)
            tfile.write(post_file_path + f": {ndcg}\n")
            tfile.flush()

if __name__ == '__main__':
   main()

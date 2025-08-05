import numpy as np
import pandas as pd

qrel_file_path = "/home/ubuntu/data_download_data/embedding_reserve/mldr_en/english/qrels/test.tsv"
query_result_file_path = "/home/ubuntu/infinity/experiments/load_experiments/MLDR_en/search_tensor_rank1000/three_roads/fulltext_dense_sparse_result.txt"

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

def main():
    qrel_map = load_qrel_map()
    df = pd.read_csv(query_result_file_path, sep='\t')
    grouped = df.groupby('query-id')
    result = {}
    for query_id, group in grouped:
        #  corpus-id 
        corpus_ids = group['corpus-id'].tolist()
        result[query_id] = corpus_ids
    result_ndcg = 0
    for query_id, corpus_id_list in result.items():
        #  query-id  qrel 
        qrel_mapping = qrel_map.get(query_id, {})
        #  query-id  corpus
        scores_list = [qrel_mapping.get(corpus_id, 0) for corpus_id in corpus_id_list]
        dcg = calculate_dcg(scores_list, 1000)
        idcg = calculate_dcg(sorted(scores_list, reverse=True), 1000)
        ndcg = dcg / idcg if idcg != 0 else 0
        print(f"Query {query_id} NDCG@10: {ndcg}")
        result_ndcg += ndcg
    print(result_ndcg / len(result))

if __name__ == '__main__':
   main()

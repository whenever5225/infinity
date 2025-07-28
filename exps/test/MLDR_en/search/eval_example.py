import math

def dcg(relevance_scores, k):
    dcg_score = relevance_scores[0]
    for i in range(1, min(k, len(relevance_scores))):
        dcg_score += relevance_scores[i] / math.log2(i + 2)
    return dcg_score

def ndcg(relevance_scores, k):
    sorted_scores = sorted(relevance_scores, reverse=True)
    idcg = dcg(sorted_scores, k)
    if idcg == 0:
        return 0
    return dcg(relevance_scores, k) / idcg

#  qrels 
qrels = {}
with open('qrel_example.txt', 'r') as f:
    for line in f:
        qid, _, docid, rel = line.strip().split()
        qid = int(qid)
        rel = int(rel)
        if qid not in qrels:
            qrels[qid] = {}
        qrels[qid][docid] = rel

#  run.txt  NDCG@10
total_ndcg = 0
num_queries = 0
with open('run_example.txt', 'r') as f:
    current_qid = None
    relevance_scores = []
    for line in f:
        qid, _, docid, _, _, _ = line.strip().split()
        qid = int(qid)
        if current_qid is None:
            current_qid = qid
        if qid != current_qid:
            total_ndcg += ndcg(relevance_scores, 10)
            num_queries += 1
            relevance_scores = []
            current_qid = qid
        if docid in qrels.get(qid, {}):
            relevance_scores.append(qrels[qid][docid])
        else:
            relevance_scores.append(0)

    # 
    total_ndcg += ndcg(relevance_scores, 10)
    num_queries += 1

#  NDCG@10
average_ndcg = total_ndcg / num_queries
print(f"Average NDCG@10: {average_ndcg}")
import threading
import time
import random
import hybridsearch
from hybridsearch.common import LOCAL_HOST

from MLDR_en.search.single_road.dense_search import dense_search
from MLDR_en.search.single_road.dense_search import GetQuestions as GetDenseQuestions
from MLDR_en.search.single_road.fulltext_search import fulltext_search
from MLDR_en.search.single_road.fulltext_search import GetQuestions as GetFulltextQuestions
from MLDR_en.search.single_road.sparse_search import sparse_search
from MLDR_en.search.single_road.sparse_search import GetQuestions as GetSparseQuestions
from MLDR_en.search.single_road.tensor_search import tensor_search
from MLDR_en.search.single_road.tensor_search import GetQuestions as GetTensorQuestions

from MLDR_en.search.two_roads.dense_fulltext_search import fulltext_dense_search
from MLDR_en.search.two_roads.dense_fulltext_search import GetQuestions as GetDenseFulltextQuestions
from MLDR_en.search.two_roads.sparse_fulltext_search import fulltext_sparse_search 
from MLDR_en.search.two_roads.sparse_fulltext_search import GetQuestions as GetSparseFulltextQuestions
from MLDR_en.search.two_roads.sparse_dense_search import dense_sparse_search
from MLDR_en.search.two_roads.sparse_dense_search import GetQuestions as GetSparseDenseQuestions
from MLDR_en.search.two_roads.sparse_tensor_search import sparse_tensor_search
from MLDR_en.search.two_roads.sparse_tensor_search import GetQuestions as GetSparseTensorQuestions
from MLDR_en.search.two_roads.dense_tensor_search import dense_tensor_search
from MLDR_en.search.two_roads.dense_tensor_search import GetQuestions as GetDenseTensorQuestions
from MLDR_en.search.two_roads.fulltext_tensor_search import fulltext_tensor_search
from MLDR_en.search.two_roads.fulltext_tensor_search import GetQuestions as GetFulltextTensorQuestions

from MLDR_en.search.three_roads.sparse_dense_fulltext_search import fulltext_dense_sparse_search
from MLDR_en.search.three_roads.sparse_dense_fulltext_search import GetQuestions as GetSparseDenseFulltextQuestions
from MLDR_en.search.three_roads.sparse_dense_tensor_search import dense_sparse_tensor_search
from MLDR_en.search.three_roads.sparse_dense_tensor_search import GetQuestions as GetSparseDenseTensorQuestions
from MLDR_en.search.three_roads.sparse_fulltext_tensor_search import fulltext_sparse_tensor_search
from MLDR_en.search.three_roads.sparse_fulltext_tensor_search import GetQuestions as GetSparseFulltextTensorQuestions
from MLDR_en.search.three_roads.dense_fulltext_tensor import fulltext_dense_tensor_search
from MLDR_en.search.three_roads.dense_fulltext_tensor import GetQuestions as GetDenseFulltextTensorQuestions

from MLDR_en.search.four_roads.sparse_fulltext_tensor_dense_search import fulltext_dense_sparse_tensor_search
from MLDR_en.search.four_roads.sparse_fulltext_tensor_dense_search import GetQuestions as GetSparseFulltextTensorDenseQUestions

funcs = [
        ## single road
        dense_search,fulltext_search,sparse_search,tensor_search,
        ## two roads
        fulltext_dense_search,fulltext_sparse_search,dense_sparse_search,sparse_tensor_search,dense_tensor_search,fulltext_tensor_search,
        ## three roads
        fulltext_dense_sparse_search,dense_sparse_tensor_search,fulltext_sparse_tensor_search,fulltext_dense_tensor_search,
        ## four roads
        fulltext_dense_sparse_tensor_search
    ]

get_questions_func = [
    ## single road
    GetDenseQuestions, GetFulltextQuestions, GetSparseQuestions, GetTensorQuestions,
    ## two roads
    GetDenseFulltextQuestions, GetSparseFulltextQuestions, GetSparseDenseQuestions, GetSparseTensorQuestions, GetDenseTensorQuestions, GetFulltextTensorQuestions,
    ## three roads
    GetSparseDenseFulltextQuestions, GetSparseDenseTensorQuestions, GetSparseFulltextTensorQuestions, GetDenseFulltextTensorQuestions,
    ## four roads
    GetSparseFulltextTensorDenseQUestions
]

process_func = dense_sparse_search
max_workers = 4
table_name = "MLDR_en_Table"

def remove_extremes_and_average(lst):
    """
    
     < 2 0

    :param lst: 
    :return: 
    """
    if len(lst) < 2:
        return 0.0  # 

    # 
    filtered = lst.copy()

    # 
    filtered.remove(max(filtered))
    filtered.remove(min(filtered))

    # 
    if not filtered:
        return 0.0

    return sum(filtered) / len(filtered)

class MultiThreadClient:
    def __init__(self,id):
        self.mt_lock = threading.Lock()
        self.queries = get_questions_func[id]()
        self.query_block = 100
        self.begin = 0
        self.table_objs = []
        self.begin_time = time.time()

    def search_main_loop(self,client_id):
        while True:
            with self.mt_lock:
                if self.begin >= len(self.queries):
                    break
                end = min(self.begin + self.query_block, len(self.queries))
                query_block = self.queries[self.begin:end]
                self.begin = end
            for query in query_block:
                process_func(self.table_objs[client_id],query)

    def setup_clients(self):
        self.clients = list()
        self.begin_time = time.time()
        for _ in range(max_workers):
            client = hybridsearch.connect(LOCAL_HOST)
            db_obj = client.get_database("default_db")
            table_obj = db_obj.get_table(table_name)
            self.clients.append(client)
            self.table_objs.append(table_obj)
        threads = []
        for i in range(max_workers):
            threads.append(
                threading.Thread(
                    target=self.search_main_loop,
                    args=[i],
                    daemon=True,
                )
            )
        for i in range(max_workers):
            threads[i].start()
         # 
        for thread in threads:
            thread.join()
        # 
        return self.post_threads_finished()

    def post_threads_finished(self):
        # 
        end_time = time.time()
        print(f"All requests processed in {(end_time - self.begin_time)*1000:.2f} ms")
        print("QPS: ", len(self.queries) / (end_time - self.begin_time))
        return len(self.queries) / (end_time - self.begin_time)

def main():
    global process_func
    with open("/home/ubuntu/hybridsearch/experiments/load_experiments/MLDR_en/search/multi_thread_result.txt",'w') as multi_thread_file:
        for id in range(len(funcs)):
            process_func = funcs[id]
            QPS = []
            for _ in range(1):
                mt_client = MultiThreadClient(id)
                QPS.append(mt_client.setup_clients())
            multi_thread_file.write(f"{process_func.__name__}"+" Average QPS: "+ str(QPS[0])+"\n")

if __name__ == "__main__":
    main()

import threading
import time
import random
import infinity
from infinity.common import LOCAL_HOST
from single_road.tensor_search_9216 import tensor_search
from single_road.tensor_search_9216 import GetQuestions as GetTensorQuestions


funcs = [
        ## single road
        tensor_search,
    ]

get_questions_func = [
    ## single road
    GetTensorQuestions,
]

process_func = tensor_search
max_workers = 4
table_name = "CQADupStack_en_Table"

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
            client = infinity.connect(LOCAL_HOST)
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
    with open("/home/ubuntu/infinity/experiments/load_experiments/CQADupStack_en/search/multi_thread_9216_result.txt",'w') as multi_thread_file:
        for id in range(len(funcs)):
            process_func = funcs[id]
            QPS = []
            for _ in range(1):
                mt_client = MultiThreadClient(id)
                QPS.append(mt_client.setup_clients())
            multi_thread_file.write(f"{process_func.__name__}"+" Average QPS: "+ str(QPS[0])+"\n")

if __name__ == "__main__":
    main()



import re
import subprocess
import time
import os
from tqdm import tqdm
import infinity
from infinity.common import ConflictType, LOCAL_HOST, SparseVector
import infinity.index as index
from infinity.errors import ErrorCode
from MLDR_zh.search_tensor_rank.three_roads.sparse_dense_fulltext_search import terminate_process_tree
from vec_read import load_dense
from vec_read import load_sparse
from colbert_read import load_colbert_list
from datasets import Dataset
## float 184601.74298286438
## float32 187938.66729736328

def extract_number(filename):
    match = re.search(r'(\d+)\.', filename)
    if match:
        return int(match.group(1))
    return 0

class InfinityClientForInsert:
    def __init__(self):
        self.test_db_name = "default_db"
        self.test_table_name_prefix = "MLDR_zh_Table"
        self.test_table_schema = {"docid_col": {"type": "varchar"}, "fulltext_col": {"type": "varchar"},
                                  "dense_col": {"type": "vector,1024,float"},
                                  "sparse_col": {"type": "sparse,250002,float,int"},
                                  "tensor_col": {"type": "tensor,64,float"}}
        self.infinity_obj = infinity.connect(LOCAL_HOST)
        self.infinity_db = self.infinity_obj.create_database(self.test_db_name, ConflictType.Ignore)
        self.infinity_table = None

    def create_test_table(self):
        table_name = self.test_table_name_prefix
        self.infinity_db.drop_table(table_name, ConflictType.Ignore)
        self.infinity_table = self.infinity_db.create_table(table_name, self.test_table_schema)
        print("Create table successfully.")

    def main(self):
        lang = 'en'
        self.create_test_table()
        corpus = Dataset.from_csv('/home/ubuntu/data_download_data/embedding_reserve/mldr_zh/mldr_zh_corpus.csv')
        total_num = corpus.num_rows
        docid_list = corpus["_id"]
        corpus_text_list = corpus["combine_text_and_title"]
        del corpus
        print(f"Expect total number of rows: {total_num}")

        dense_embedding_dir = '/home/ubuntu/data_download_data/embedding_reserve/MLDR_zh/dense_embeddings/vectors'

        sparse_embedding_dir = '/home/ubuntu/data_download_data/embedding_reserve/MLDR_zh/sparse_embeddings/vectors'
        tensor_embedding_dir = '/home/ubuntu/large_disk/infinity/mldr_zh/tensor_embeddings/vectors'
        dense_names = [f for f in os.listdir(dense_embedding_dir) if os.path.isfile(os.path.join(dense_embedding_dir, f))]
        sparse_names = [f for f in os.listdir(sparse_embedding_dir) if os.path.isfile(os.path.join(sparse_embedding_dir, f))]
        tensor_names = [f for f in os.listdir(tensor_embedding_dir) if os.path.isfile(os.path.join(tensor_embedding_dir, f))]
        dense_names = sorted(dense_names, key=extract_number)
        sparse_names = sorted(sparse_names, key=extract_number)
        tensor_names = sorted(tensor_names, key=extract_number)
        print("Start inserting data...")
        dense_data = None

        sparse_data = None

        dense_file_idx = 0
        sparse_file_idx = 0
        tensor_file_idx = 0
        doc_id_idx = 0
        dense_idx = 0
        sparse_idx = 0
        tensor_idx = 0

        sparse_nums = 0
        tensor_nums = 0
        dense_nums = 0
        begin_insert_time = time.time()
        buffer = []
        while dense_file_idx < len(dense_names) or sparse_file_idx < len(sparse_names) or tensor_file_idx < len(tensor_names):
            dense_path = os.path.join(dense_embedding_dir,dense_names[dense_file_idx])
            sparse_path = os.path.join(sparse_embedding_dir,sparse_names[sparse_file_idx])
            tensor_path = os.path.join(tensor_embedding_dir,tensor_names[tensor_file_idx])
            dense_vectors = load_dense(dense_path)
            sparse_vectors = load_sparse(sparse_path)
            tensor_vectors = load_colbert_list(tensor_path)
            
            while dense_idx < len(dense_vectors) and sparse_idx < len(sparse_vectors) and tensor_idx < len(tensor_vectors):
                indices = []
                values = []

                sparse_nums += 1
                tensor_nums += 1
                dense_nums += 1
                for key, value in sparse_vectors[sparse_idx].items():
                    indices.append(int(key))
                    values.append(value)
                insert_dict = {
                                "docid_col": docid_list[doc_id_idx], "fulltext_col": corpus_text_list[doc_id_idx],
                                "dense_col": dense_vectors[dense_idx], "sparse_col": SparseVector(indices,values),
                                "tensor_col": tensor_vectors[tensor_idx],
                               }
                doc_id_idx += 1
                dense_idx += 1
                sparse_idx += 1
                tensor_idx += 1
                buffer.append(insert_dict)
            while len(buffer) >= 500:
                self.infinity_table.insert(buffer[:500])
                buffer = buffer[500:]
            if dense_idx >= len(dense_vectors):
                dense_idx = 0
                dense_file_idx += 1
            if sparse_idx >= len(sparse_vectors):
                sparse_idx = 0
                sparse_file_idx += 1
            if tensor_idx >= len(tensor_vectors):
                tensor_idx = 0
                tensor_file_idx += 1
        if len(buffer) > 0:
            self.infinity_table.insert(buffer)
        end_insert_time = time.time()
        print("insert time: ",(end_insert_time - begin_insert_time)*1000,'ms')
        current_dir = os.path.dirname(os.path.abspath(__file__))
        current_file_path = __file__
        current_file_name = os.path.basename(current_file_path)
        current_file_name_without_extension = os.path.splitext(current_file_name)[0]
        with open(current_dir + "/" + current_file_name_without_extension + ".time",'w') as tfile:
            tfile.write(f"{(end_insert_time - begin_insert_time)*1000} ms")
            tfile.flush()
        print(f"Finish inserting data. tensor_nums: {tensor_nums+1}, sparse_nums: {sparse_nums+1}, dense_nums: {dense_nums+1}, text_nums: {doc_id_idx+1}")
        del dense_data
        del sparse_data
        del docid_list
        del corpus_text_list

if __name__ == "__main__":
    time.sleep(3)
    # 
    service_command = "/home/ubuntu/infinity/cmake-build-release/src/infinity -f /home/ubuntu/infinity/conf/infinity_conf.toml"  #  HTTP 
    process = subprocess.Popen(service_command, shell=True)
    time.sleep(3)
    print(f" ID: {process.pid}")
    print(__file__)
    infinity_client = InfinityClientForInsert()
    infinity_client.main()
    terminate_process_tree(process.pid)
    print(f" {process.pid} ")



import re
import time
import os
from tqdm import tqdm
# from mldr_common_tools import load_corpus, fvecs_read_yield, read_mldr_sparse_embedding_yield, get_all_part_begin_ends
import hybridsearch
from hybridsearch.common import ConflictType, LOCAL_HOST, SparseVector
import hybridsearch.index as index
from hybridsearch.errors import ErrorCode
from vec_read import load_dense
from vec_read import load_sparse
from colbert_read import load_colbert_list
from datasets import Dataset

import subprocess
import time

import psutil
def terminate_process_tree(pid):
    try:
        parent = psutil.Process(pid)
        for child in parent.children(recursive=True):
            child.terminate()
        parent.terminate()
    except psutil.NoSuchProcess:
        pass

def read_file_content(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            return content
    except FileNotFoundError:
        print(f" {file_path} ")
    except Exception as e:
        print(f" - {e}")
    return None

def extract_number(filename):
    match = re.search(r'(\d+)\.', filename)
    if match:
        return int(match.group(1))
    return 0

class hybridsearchClientForInsert:
    def __init__(self):
        self.test_db_name = "default_db"
        self.test_table_name_prefix = "mldr_en_Table"
        self.test_table_schema = {"docid_col": {"type": "varchar"}, "fulltext_col": {"type": "varchar"},
                                  "dense_col": {"type": "vector,1024,float"},
                                  "sparse_col": {"type": "sparse,250002,float,int"},
                                  "tensor_col": {"type": "tensor,96,float"}}
        self.hybridsearch_obj = hybridsearch.connect(LOCAL_HOST)
        self.hybridsearch_db = self.hybridsearch_obj.create_database(self.test_db_name, ConflictType.Ignore)
        self.hybridsearch_table = None

    def create_test_table(self):
        table_name = self.test_table_name_prefix
        self.hybridsearch_db.drop_table(table_name, ConflictType.Ignore)
        self.hybridsearch_table = self.hybridsearch_db.create_table(table_name, self.test_table_schema)
        print("Create table successfully.")

    def main(self):
        lang = 'en'
        self.create_test_table()
        corpus = Dataset.from_csv('/home/ubuntu/data_download_data/embedding_reserve/mldr_en/mldr_en_corpus.csv')
        total_num = corpus.num_rows
        docid_list = corpus["docid"]
        corpus_text_list = corpus["text"]
        del corpus
        print(f"Expect total number of rows: {total_num}")

        dense_embedding_dir = '/home/ubuntu/data_download_data/embedding_reserve/mldr_en/dense_embeddings/vectors'

        sparse_embedding_dir = '/home/ubuntu/data_download_data/embedding_reserve/mldr_en/sparse_embedding/vectors'
        tensor_embedding_dir = '/home/ubuntu/data_download_data/embedding_reserve/mldr_en/tensor_embeddings/vectors'
        dense_names = [f for f in os.listdir(dense_embedding_dir) if os.path.isfile(os.path.join(dense_embedding_dir, f))]
        sparse_names = [f for f in os.listdir(sparse_embedding_dir) if os.path.isfile(os.path.join(sparse_embedding_dir, f))]
        tensor_names = [f for f in os.listdir(tensor_embedding_dir) if os.path.isfile(os.path.join(tensor_embedding_dir, f))]
        dense_names = sorted(dense_names, key=extract_number)
        sparse_names = sorted(sparse_names, key=extract_number) 
        tensor_names = sorted(tensor_names, key=extract_number)
        insert_num = total_num

        batch_size = insert_num/10
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
                self.hybridsearch_table.insert(buffer[:500])
                buffer = buffer[500:]
            print(f"DocID: {doc_id_idx}")
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
            self.hybridsearch_table.insert(buffer)
        end_insert_time = time.time()
        print("insert time: ",(end_insert_time - begin_insert_time)*1000,'ms')

        print(f"Finish inserting data. tensor_nums: {tensor_nums+1}, sparse_nums: {sparse_nums+1}, dense_nums: {dense_nums+1}, text_nums: {doc_id_idx+1}")
        del dense_data
        del sparse_data
        del docid_list
        del corpus_text_list

if __name__ == "__main__":
    hybridsearch_client = hybridsearchClientForInsert()
    hybridsearch_client.main()

# Copyright(C) 2023 InfiniFlow, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This example is to connect local infinity instance, create table, insert data, search the data
"""

# import infinity_embedded as infinity
import signal
from multi_client import use_multi_client
import os
import re
import time
import infinity
import sys
from infinity.common import SparseVector
from infinity.common import LOCAL_HOST
import pandas as pd
from SciFact_en.search.colbert_read import load_colbert_list
from utils import add_escape_characters
from SciFact_en.search.vec_read import load_dense, load_sparse

path_prefix = "/home/ubuntu/infinity/experiments/load_experiments/SciFact_en/search_tensor_rank"

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

cost_time = 0
def fulltext_dense_sparse_search(infinity_table, question):
     global cost_time
     begin_time = time.time()
     res = (
                infinity_table.output(["docid_col","fulltext_col"])
                .match_text("fulltext_col", question[0], 10)
                .match_dense("dense_col", question[1], "float", "ip", 10, {"ef": "600"})
                .match_sparse("sparse_col", question[2], "ip", 10)
                # .match_tensor("tensor_col", question[2], "float", 10)
                # .fusion(method="rrf",topn=10)
                .fusion(
                    method="match_tensor", topn=10,
                    fusion_params={"field": "tensor_col", "element_type": "float",
                                "query_tensor": question[-1]}
                )
                # .to_pl()
            )
     end_time = time.time()
     if use_multi_client == True:
         return None,None
     cost_time += (end_time-begin_time)*1000
     qb_result, extra_result = res.to_pl()
     return qb_result, extra_result

def GetAllQuestions():
    df = pd.read_csv('/home/ubuntu/infinity/experiments/load_experiments/SciFact_en/queries_with_id.csv')
    query_ids = []
    fulltext_questions = []
    dense_questions = []
    sparse_questions = []
    tensor_questions = []
    ## rowids
    for _, row in df.iterrows():
        query_ids.append(row['_id'])
    ## fulltext
    for _, row in df.iterrows():
        fulltext_questions.append(add_escape_characters(row['text']).strip())
    ## dense
    dense_embedding_dir = "/home/ubuntu/data_download_data/embedding_reserve/SciFact_en/query_dense_embeddings/vectors"
    dense_names = [f for f in os.listdir(dense_embedding_dir) if os.path.isfile(os.path.join(dense_embedding_dir, f))]
    dense_names = sorted(dense_names, key=extract_number)
    dense_file_idx = 0
    id = 0
    while dense_file_idx < len(dense_names):
        dense_path = os.path.join(dense_embedding_dir,dense_names[dense_file_idx])
        query_dense_vectors = load_dense(dense_path)
        for i in range(len(query_dense_vectors)):
            dense_questions.append(query_dense_vectors[i])
            id += 1
        dense_file_idx += 1
    ## sparse
    sparse_embedding_dir = "/home/ubuntu/data_download_data/embedding_reserve/SciFact_en/query_sparse_embeddings/vectors"
    sparse_names = [f for f in os.listdir(sparse_embedding_dir) if os.path.isfile(os.path.join(sparse_embedding_dir, f))]
    sparse_names = sorted(sparse_names, key=extract_number)
    sparse_file_idx = 0
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
            sparse_questions.append(SparseVector(indices,values))
            id += 1
        sparse_file_idx += 1
    ## tensor     
    tensor_embedding_dir = "/home/ubuntu/data_download_data/embedding_reserve/SciFact_en/query_tensor_embeddings/vectors"
    tensor_names = [f for f in os.listdir(tensor_embedding_dir) if os.path.isfile(os.path.join(tensor_embedding_dir, f))]
    tensor_names = sorted(tensor_names,key=extract_number)
    tensor_file_idx = 0
    id = 0
    while tensor_file_idx < len(tensor_names):
        tensor_path = os.path.join(tensor_embedding_dir,tensor_names[tensor_file_idx])
        query_tensor_vectors = load_colbert_list(tensor_path)
        for i in range(len(query_tensor_vectors)):
            tensor_questions.append(query_tensor_vectors[i])
            id += 1
        tensor_file_idx += 1
    return (query_ids, fulltext_questions, dense_questions, sparse_questions, tensor_questions)

def GetQuestions():
    query_ids, fulltext_questions, dense_questions, sparse_questions, tensor_questions = GetAllQuestions()
    questions = []
    for i in range(len(query_ids)):
        questions.append((fulltext_questions[i],dense_questions[i],sparse_questions[i],query_ids[i],tensor_questions[i]))
    print(len(questions))
    return questions

def single_search(questions):
    try:
        #  use infinity module to connect a remote server
        infinity_instance = infinity.connect(LOCAL_HOST)

        # 'default_db' is the default database
        db_instance = infinity_instance.get_database("default_db")
        infinity_table = db_instance.get_table("SciFact_en_Table")
        with open(path_prefix + '/three_roads/fulltext_dense_sparse_result.txt','w') as result_file:
            id = 0
            time_cost = 0
            result_file.write("query-id\tcorpus-id\n")
            for question in questions:
                id += 1
                begin_time = time.time()
                qb_result, extra_result = fulltext_dense_sparse_search(infinity_table, question)
                end_time = time.time()
                time_cost += (end_time - begin_time) * 1000
                for i in range(len(qb_result['docid_col'])):
                    result_file.write(f"{question[3]}\t{qb_result['docid_col'][i]}\n")
                result_file.flush()
                if extra_result is not None:
                    print(extra_result)
            time_cost = time_cost / len(questions)
            print(f"time_cost: {time_cost} ms")
        infinity_instance.disconnect()
        return time_cost
    except Exception as e:
        print(str(e))
        sys.exit(-1)

if __name__ == "__main__":
    # 
    service_command = "/home/ubuntu/infinity/cmake-build-release/src/infinity -f /home/ubuntu/infinity/conf/infinity_conf.toml"  #  HTTP 
    process = subprocess.Popen(service_command, shell=True)
    time.sleep(3)
    print(f" ID: {process.pid}")
    print(__file__)
    time_cost = []
    questions = GetQuestions()
    for _ in range(1):
        time_cost.append(single_search(questions))
    print("Average Time Cost: ", remove_extremes_and_average(time_cost)," ms")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    current_file_path = __file__
    current_file_name = os.path.basename(current_file_path)
    current_file_name_without_extension = os.path.splitext(current_file_name)[0]
    with open(current_dir + "/" + current_file_name_without_extension + ".time",'w') as tfile:
        tfile.write(f"{cost_time/len(questions)} ms")
        tfile.flush()
    # 
    file_path = '/home/ubuntu/infinity/experiments/query_memory_file'
    content = read_file_content(file_path)
    with open(current_dir + "/" + current_file_name_without_extension + ".memory",'w') as mfile:
        mfile.write(content)
        mfile.flush()
    terminate_process_tree(process.pid)
    print(f" {process.pid} ")

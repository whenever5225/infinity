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
from multi_client import use_multi_client
import os
import re
import time
from Touche_2020_en.search.vec_read import load_sparse
import infinity
import sys
from infinity.common import SparseVector
from infinity.common import LOCAL_HOST
import pandas as pd
from utils import add_escape_characters

path_prefix = "/home/ubuntu/infinity/experiments/load_experiments/Touche_2020_en/search"

def extract_number(filename):
    match = re.search(r'(\d+)\.', filename)
    if match:
        return int(match.group(1))
    return 0

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

cost_time = 0
def sparse_search(infinity_table, question):
     global cost_time
     begin_time = time.time()
     res = (
                infinity_table.output(["docid_col","sparse_col"])
                .match_sparse("sparse_col", question[0], "ip", 10)
                # .to_pl()
            )
     end_time = time.time()
     if use_multi_client == True:
         return None,None
     cost_time += (end_time-begin_time)*1000
     qb_result, extra_result = res.to_pl()
     return qb_result, extra_result

def GetQuestions():
    sparse_embedding_dir = "/home/ubuntu/data_download_data/embedding_reserve/Touche_2020_en/query_sparse_embeddings/vectors"
    df = pd.read_csv('/home/ubuntu/infinity/experiments/load_experiments/Touche_2020_en/queries_with_id.csv')
    sparse_names = [f for f in os.listdir(sparse_embedding_dir) if os.path.isfile(os.path.join(sparse_embedding_dir, f))]
    sparse_names = sorted(sparse_names,key=extract_number)
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
            questions.append((SparseVector(indices,values),df.iloc[id]['_id']))
            id += 1
        sparse_file_idx += 1
    print("questions: ",len(questions))
    return questions

def single_search(questions):
    try:
        #  use infinity module to connect a remote server
        infinity_instance = infinity.connect(LOCAL_HOST)

        # 'default_db' is the default database
        db_instance = infinity_instance.get_database("default_db")
        infinity_table = db_instance.get_table("Touche_2020_en_Table")
        with open(path_prefix + '/single_road/sparse_result.txt','w') as result_file:
            id = 0
            time_cost = 0
            result_file.write("query-id\tcorpus-id\n")
            for question in questions:
                id += 1
                begin_time = time.time()
                qb_result, extra_result = sparse_search(infinity_table, question)
                end_time = time.time()
                time_cost += (end_time - begin_time) * 1000
                for i in range(len(qb_result['docid_col'])):
                    result_file.write(f"{question[1]}\t{qb_result['docid_col'][i]}\n")
                result_file.flush()
                if extra_result is not None:
                    print(extra_result)
            print("lastid: ",id)
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

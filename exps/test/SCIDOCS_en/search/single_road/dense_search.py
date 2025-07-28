# Copyright(C) 2023 HybridSearchFlow, Inc. All rights reserved.
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
This example is to connect local hybridsearch instance, create table, insert data, search the data
"""

# import hybridsearch_embedded as hybridsearch
from multi_client import use_multi_client
import os
import re
import time 
from SCIDOCS_en.search.vec_read import load_dense
import hybridsearch
import sys

from hybridsearch.common import LOCAL_HOST
import pandas as pd
from utils import add_escape_characters

def extract_number(filename):
    match = re.search(r'(\d+)\.', filename)
    if match:
        return int(match.group(1))
    return 0

path_prefix = "/home/ubuntu/hybridsearch/experiments/load_experiments/SCIDOCS_en/search"

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
def dense_search(hybridsearch_table, question):
     global cost_time
     begin_time = time.time()
     res = (
                hybridsearch_table.output(["docid_col","dense_col"])
                .match_dense("dense_col", question[0], "float", "ip", 10, {"ef": "600"})
                # .to_pl()
            )
     end_time = time.time()
     if use_multi_client == True:
         return None,None
     cost_time += (end_time-begin_time)*1000
     qb_result, extra_result = res.to_pl()
     return qb_result, extra_result

def GetQuestions():
    dense_embedding_dir = "/home/ubuntu/data_download_data/embedding_reserve/SCIDOCS_en/query_dense_embeddings/vectors"
    df = pd.read_csv('/home/ubuntu/hybridsearch/experiments/load_experiments/SCIDOCS_en/queries_with_id.csv')
    dense_names = [f for f in os.listdir(dense_embedding_dir) if os.path.isfile(os.path.join(dense_embedding_dir, f))]
    dense_names = sorted(dense_names,key=extract_number)
    dense_file_idx = 0
    questions = []
    id = 0
    while dense_file_idx < len(dense_names):
        dense_path = os.path.join(dense_embedding_dir,dense_names[dense_file_idx])
        query_dense_vectors = load_dense(dense_path)
        for i in range(len(query_dense_vectors)):
            questions.append((query_dense_vectors[i],df.iloc[id]['_id']))
            id += 1
        dense_file_idx += 1
    return questions

def single_search(questions):
    try:
        #  Use hybridsearch module to connect a remote server
        hybridsearch_instance = hybridsearch.connect(LOCAL_HOST)

        # 'default_db' is the default database
        db_instance = hybridsearch_instance.get_database("default_db")
        hybridsearch_table = db_instance.get_table("SCIDOCS_en_Table")
        with open(path_prefix + '/single_road/dense_result.txt','w') as result_file:
            id = 0
            time_cost = 0
            result_file.write("query-id\tcorpus-id\n")
            for question in questions:
                id += 1
                begin_time = time.time()
                qb_result, extra_result = dense_search(hybridsearch_table, question)
                end_time = time.time()
                time_cost += (end_time - begin_time) * 1000
                for i in range(len(qb_result['docid_col'])):
                    result_file.write(f"{question[1]}\t{qb_result['docid_col'][i]}\n")
                result_file.flush()
                if extra_result is not None:
                    print(extra_result)
            time_cost = time_cost / len(questions)
            print(f"time_cost: {time_cost} ms")
        hybridsearch_instance.disconnect()
        return time_cost
    except Exception as e:
        print(str(e))
        sys.exit(-1)

if __name__ == "__main__":
    # 
    service_command = "/home/ubuntu/hybridsearch/cmake-build-release/src/hybridsearch -f /home/ubuntu/hybridsearch/conf/hybridsearch_conf.toml"  #  HTTP 
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
    file_path = '/home/ubuntu/hybridsearch/experiments/query_memory_file'
    content = read_file_content(file_path)
    with open(current_dir + "/" + current_file_name_without_extension + ".memory",'w') as mfile:
        mfile.write(content)
        mfile.flush()
    terminate_process_tree(process.pid)
    print(f" {process.pid} ")
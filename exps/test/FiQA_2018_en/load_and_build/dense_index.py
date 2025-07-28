

"""
This example is to connect local hybridsearch instance, create table, insert data, search the data
"""

# import hybridsearch_embedded as hybridsearch
import os
import subprocess
from multi_client import use_multi_client
import time
import hybridsearch
import sys
import hybridsearch.index as index
from hybridsearch.errors import ErrorCode
from hybridsearch.common import ConflictType, LOCAL_HOST, SparseVector

from FiQA_2018_en.search_tensor_rank.three_roads.sparse_dense_fulltext_search import read_file_content, terminate_process_tree

try:
    # 
    service_command = "/home/ubuntu/hybridsearch/cmake-build-release/src/hybridsearch -f /home/ubuntu/hybridsearch/conf/hybridsearch_conf.toml"  #  HTTP 
    process = subprocess.Popen(service_command, shell=True)
    time.sleep(3)
    print(f" ID: {process.pid}")
    print(__file__)
    #  Use hybridsearch module to connect a remote server
    hybridsearch_instance = hybridsearch.connect(LOCAL_HOST)
    lang = 'en'
    ft_params = None
    # print("Start creating Hnsw index.")
    hybridsearch_db = hybridsearch_instance.get_database('default_db')
    hybridsearch_table = hybridsearch_db.get_table("FiQA_2018_en_Table")
    hybridsearch_table.drop_index('FiQA_2018_en_hnsw_index',ConflictType.Ignore)
    print("Start creating Hnsw index...")
    begin_hnsw_time = time.time()
    res = hybridsearch_table.create_index("FiQA_2018_en_hnsw_index", index.IndexInfo("dense_col", index.IndexType.Hnsw,
                                                                            {
                                                                                "m": "16",
                                                                                "ef_construction": "200",
                                                                                "metric": "ip",
                                                                                "encode": "lvq"
                                                                            }),
                                            ConflictType.Error)
    end_hnsw_time = time.time()
    hybridsearch_table.show_index("FiQA_2018_en_hnsw_index")
    print("end hnsw time: ",(end_hnsw_time - begin_hnsw_time)*1000,'ms')
    print("Finish creating Hnsw index.")
    assert res.error_code == ErrorCode.OK

    cost_time = (end_hnsw_time - begin_hnsw_time)*1000
    current_dir = os.path.dirname(os.path.abspath(__file__))
    current_file_path = __file__
    current_file_name = os.path.basename(current_file_path)
    current_file_name_without_extension = os.path.splitext(current_file_name)[0]
    with open(current_dir + "/" + current_file_name_without_extension + ".time",'w') as tfile:
        tfile.write(f"{cost_time} ms")
        tfile.flush()
    # 
    file_path = '/home/ubuntu/hybridsearch/experiments/peak_memory_index_file'
    content = read_file_content(file_path)
    with open(current_dir + "/" + current_file_name_without_extension + ".memory",'w') as mfile:
        mfile.write(content)
        mfile.flush()

    terminate_process_tree(process.pid)
    print(f" {process.pid} ")
    print('build done')
    sys.exit(0)
except Exception as e:
    print(str(e))
    sys.exit(-1)

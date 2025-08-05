

"""
This example is to connect local infinity instance, create table, insert data, search the data
"""

# import infinity_embedded as infinity
import os
import signal
import subprocess
from multi_client import use_multi_client
import time
import infinity
import sys
import infinity.index as index
from infinity.errors import ErrorCode
from infinity.common import ConflictType, LOCAL_HOST, SparseVector

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

try:
    # 
    service_command = "/home/ubuntu/infinity/cmake-build-release/src/infinity -f /home/ubuntu/infinity/conf/infinity_conf.toml"  #  HTTP 
    process = subprocess.Popen(service_command, shell=True)
    time.sleep(3)
    print(f" ID: {process.pid}")
    print(__file__)
    #  use infinity module to connect a remote server
    infinity_instance = infinity.connect(LOCAL_HOST)
    lang = 'en'
    ft_params = None
    # print("Start creating Hnsw index.")
    infinity_db = infinity_instance.get_database('default_db')
    infinity_table = infinity_db.get_table("mldr_en_Table")
    infinity_table.drop_index('mldr_en_tensor_index',ConflictType.Ignore)
    print("Start creating tensor index...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    current_file_path = __file__
    current_file_name = os.path.basename(current_file_path)
    current_file_name_without_extension = os.path.splitext(current_file_name)[0]
    begin_hnsw_time = time.time()
    res = infinity_table.create_index("mldr_en_tensor_index", index.IndexInfo("tensor_col", index.IndexType.EMVB,
                                    {
                                        "pq_subspace_num":"32",
                                        "pq_subspace_bits":"8",
                                    }
                                ),
                            ConflictType.Error)
    end_hnsw_time = time.time()
    with open(current_dir + "/" + current_file_name_without_extension + ".time",'w') as tfile:
        tfile.write(f"{(end_hnsw_time-begin_hnsw_time)*1000} ms")
        tfile.flush()
    # 
    file_path = '/home/ubuntu/infinity/experiments/peak_memory_index_file'
    content = read_file_content(file_path)
    with open(current_dir + "/" + current_file_name_without_extension + ".memory",'w') as mfile:
        mfile.write(content)
        mfile.flush()
    terminate_process_tree(process.pid)
    print(f" {process.pid} ")
    print(infinity_table.show_index("mldr_en_tensor_index"))
    print("end tensor time: ",(end_hnsw_time - begin_hnsw_time)*1000,'ms')
    print("Finish creating tensor index.")
    assert res.error_code == ErrorCode.OK

    print('build done')
    sys.exit(0)
except Exception as e:
    print(str(e))
    sys.exit(-1)

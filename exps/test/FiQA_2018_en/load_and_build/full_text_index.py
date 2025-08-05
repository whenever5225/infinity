

"""
This example is to connect local infinity instance, create table, insert data, search the data
"""

# import infinity_embedded as infinity
import os
import subprocess
from multi_client import use_multi_client
import time
import infinity
import sys
import infinity.index as index
from infinity.errors import ErrorCode
from infinity.common import ConflictType, LOCAL_HOST, SparseVector

from FiQA_2018_en.search_tensor_rank.three_roads.sparse_dense_fulltext_search import read_file_content, terminate_process_tree

try:
    # 
    service_command = "/home/ubuntu/infinity/cmake-build-release/src/infinity -f /home/ubuntu/infinity/conf/infinity_conf.toml"  #  HTTP 
    process = subprocess.Popen(service_command, shell=True)
    time.sleep(3)
    print(f" ID: {process.pid}")
    print(__file__)
    #  use infinity modul
    #  use infinity module to connect a remote server
    infinity_instance = infinity.connect(LOCAL_HOST)
    lang = 'en'
    ft_params = None
    print("Start creating fulltext index.")
    if lang == "zh":
        ft_params = {"ANALYZER": "chinese"}
    
    infinity_db = infinity_instance.get_database('default_db')
    infinity_table = infinity_db.get_table("FiQA_2018_en_Table")
    infinity_table.drop_index('FiQA_2018_en_ft_index',ConflictType.Ignore)
    begin_full_text_time = time.time()
    res = infinity_table.create_index("FiQA_2018_en_ft_index",
                                            index.IndexInfo("fulltext_col", index.IndexType.FullText, ft_params),
                                            ConflictType.Error)
    infinity_table.show_index("FiQA_2018_en_ft_index")
    end_full_text_time = time.time()
    print(infinity_table.show_index('FiQA_2018_en_ft_index'))
    print("end full text time: ",(end_full_text_time - begin_full_text_time)*1000,'ms')
    assert res.error_code == ErrorCode.OK

    cost_time = (end_full_text_time - begin_full_text_time)*1000
    current_dir = os.path.dirname(os.path.abspath(__file__))
    current_file_path = __file__
    current_file_name = os.path.basename(current_file_path)
    current_file_name_without_extension = os.path.splitext(current_file_name)[0]
    with open(current_dir + "/" + current_file_name_without_extension + ".time",'w') as tfile:
        tfile.write(f"{cost_time} ms")
        tfile.flush()
    # 
    file_path = '/home/ubuntu/infinity/experiments/peak_memory_index_file'
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

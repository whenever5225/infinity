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
import os
import subprocess
from multi_client import use_multi_client
import time
import infinity
import sys
import infinity.index as index
from infinity.errors import ErrorCode
from infinity.common import ConflictType, LOCAL_HOST, SparseVector

from TREC_COVID_en.search_tensor_rank.three_roads.sparse_dense_fulltext_search import read_file_content, terminate_process_tree

try:
    # 启动服务并在适当的时候杀死它
    service_command = "/home/ubuntu/infinity/cmake-build-release/src/infinity -f /home/ubuntu/infinity/conf/infinity_conf.toml"  # 示例命令，启动一个简单的 HTTP 服务器
    process = subprocess.Popen(service_command, shell=True)
    time.sleep(3)
    print(f"服务已启动，进程 ID: {process.pid}")
    print(__file__)
    #  use infinity module to connect a remote server
    infinity_instance = infinity.connect(LOCAL_HOST)
    lang = 'en'
    ft_params = None
    # print("Start creating Hnsw index.")
    infinity_db = infinity_instance.get_database('default_db')
    infinity_table = infinity_db.get_table("TREC_COVID_en_Table")
    infinity_table.drop_index('TREC_COVID_en_bmp_index',ConflictType.Ignore)
    print("Start creating Bmp index...")
    begin_hnsw_time = time.time()
    res = infinity_table.create_index("TREC_COVID_en_bmp_index", index.IndexInfo("sparse_col", index.IndexType.BMP,
                                                                            {
                                                                                "block_size": "8",
                                                                                "compress_type": "compress",
                                                                                # "metric": "ip",
                                                                            }),
                                               ConflictType.Error)
    infinity_table.optimize("TREC_COVID_en_bmp_index", {"topk": "1000"})
    end_hnsw_time = time.time()
    infinity_table.show_index("TREC_COVID_en_bmp_index")
    print("end Bmp time: ",(end_hnsw_time - begin_hnsw_time)*1000,'ms')
    print("Finish creating BMP index.")
    assert res.error_code == ErrorCode.OK

    cost_time = (end_hnsw_time - begin_hnsw_time)*1000
    current_dir = os.path.dirname(os.path.abspath(__file__))
    current_file_path = __file__
    current_file_name = os.path.basename(current_file_path)
    current_file_name_without_extension = os.path.splitext(current_file_name)[0]
    with open(current_dir + "/" + current_file_name_without_extension + ".time",'w') as tfile:
        tfile.write(f"{cost_time} ms")
        tfile.flush()
    # 读取文件内容
    file_path = '/home/ubuntu/infinity/experiments/peak_memory_index_file'
    content = read_file_content(file_path)
    with open(current_dir + "/" + current_file_name_without_extension + ".memory",'w') as mfile:
        mfile.write(content)
        mfile.flush()

    terminate_process_tree(process.pid)
    print(f"服务进程 {process.pid} 已被终止。")
    print('build done')
    sys.exit(0)
except Exception as e:
    print(str(e))
    sys.exit(-1)

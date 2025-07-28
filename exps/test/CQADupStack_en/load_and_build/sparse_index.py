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
import time
import hybridsearch
import sys
import hybridsearch.index as index
from hybridsearch.errors import ErrorCode
from hybridsearch.common import ConflictType, LOCAL_HOST, SparseVector

try:
    #  Use hybridsearch module to connect a remote server
    hybridsearch_instance = hybridsearch.connect(LOCAL_HOST)
    lang = 'en'
    ft_params = None
    # print("Start creating Hnsw index.")
    hybridsearch_db = hybridsearch_instance.get_database('default_db')
    hybridsearch_table = hybridsearch_db.get_table("CQADupStack_en_Table")
    hybridsearch_table.drop_index('CQADupStack_en_bmp_index',ConflictType.Ignore)
    print("Start creating Bmp index...")
    begin_hnsw_time = time.time()
    res = hybridsearch_table.create_index("CQADupStack_en_bmp_index", index.IndexInfo("sparse_col", index.IndexType.BMP,
                                                                            {
                                                                                "block_size": "8",
                                                                                "compress_type": "compress",
                                                                                # "metric": "ip",
                                                                            }),
                                               ConflictType.Error)
    hybridsearch_table.optimize("CQADupStack_en_bmp_index", {"topk": "1000"})
    end_hnsw_time = time.time()
    hybridsearch_table.show_index("CQADupStack_en_bmp_index")
    print("end Bmp time: ",(end_hnsw_time - begin_hnsw_time)*1000,'ms')
    print("Finish creating BMP index.")
    assert res.error_code == ErrorCode.OK

    print('build done')
    sys.exit(0)
except Exception as e:
    print(str(e))
    sys.exit(-1)

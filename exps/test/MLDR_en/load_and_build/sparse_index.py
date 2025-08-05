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
import time
import infinity
import sys
import infinity.index as index
from infinity.errors import ErrorCode
from infinity.common import ConflictType, LOCAL_HOST, SparseVector

try:
    #  use infinity module to connect a remote server
    infinity_instance = infinity.connect(LOCAL_HOST)
    lang = 'en'
    ft_params = None
    # print("Start creating Hnsw index.")
    infinity_db = infinity_instance.get_database('default_db')
    infinity_table = infinity_db.get_table("mldr_en_Table")
    infinity_table.drop_index('mldr_en_bmp_index',ConflictType.Ignore)
    print("Start creating Bmp index...")
    begin_hnsw_time = time.time()
    res = infinity_table.create_index("mldr_en_bmp_index", index.IndexInfo("sparse_col", index.IndexType.BMP,
                                                                            {
                                                                                "block_size": "8",
                                                                                "compress_type": "compress",
                                                                                # "metric": "ip",
                                                                            }),
                                               ConflictType.Error)
    infinity_table.optimize("mldr_en_bmp_index", {"topk": "1000"})
    end_hnsw_time = time.time()
    print(infinity_table.show_index("mldr_en_bmp_index"))
    print("end Bmp time: ",(end_hnsw_time - begin_hnsw_time)*1000,'ms')
    print("Finish creating BMP index.")
    assert res.error_code == ErrorCode.OK

    print('build done')
    sys.exit(0)
except Exception as e:
    print(str(e))
    sys.exit(-1)

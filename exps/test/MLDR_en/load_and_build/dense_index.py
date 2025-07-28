

"""
This example is to connect local hybridsearch instance, create table, insert data, search the data
"""

# import hybridsearch_embedded as hybridsearch
import signal
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
    hybridsearch_table = hybridsearch_db.get_table("mldr_en_Table")
    hybridsearch_table.drop_index('mldr_en_hnsw_index',ConflictType.Ignore)
    print("Start creating Hnsw index...")
    begin_hnsw_time = time.time()
    res = hybridsearch_table.create_index("mldr_en_hnsw_index", index.IndexInfo("dense_col", index.IndexType.Hnsw,
                                                                            {
                                                                                "m": "16",
                                                                                "ef_construction": "200",
                                                                                "metric": "ip",
                                                                                "encode": "lvq"
                                                                            }),
                                            ConflictType.Error)
    end_hnsw_time = time.time()
    print(hybridsearch_table.show_index("mldr_en_hnsw_index"))
    print("end hnsw time: ",(end_hnsw_time - begin_hnsw_time)*1000,'ms')
    print("Finish creating Hnsw index.")
    assert res.error_code == ErrorCode.OK

    print('build done')
    sys.exit(0)
except Exception as e:
    print(str(e))
    sys.exit(-1)

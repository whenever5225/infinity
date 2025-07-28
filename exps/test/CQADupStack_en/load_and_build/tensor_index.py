

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
    hybridsearch_table.drop_index('CQADupStack_en_tensor_index',ConflictType.Ignore)
    print("Start creating tensor index...")
    begin_hnsw_time = time.time()
    res = hybridsearch_table.create_index("CQADupStack_en_tensor_index", index.IndexInfo("tensor_col", index.IndexType.EMVB,
                                    {
                                        "pq_subspace_num":"32",
                                        "pq_subspace_bits":"8",
                                    }
                                ),
                            ConflictType.Error)
    end_hnsw_time = time.time()
    hybridsearch_table.show_index("CQADupStack_en_tensor_index")
    print("end tensor time: ",(end_hnsw_time - begin_hnsw_time)*1000,'ms')
    print("Finish creating tensor index.")
    assert res.error_code == ErrorCode.OK

    print('build done')
    sys.exit(0)
except Exception as e:
    print(str(e))
    sys.exit(-1)

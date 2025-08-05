

"""
This example is to connect local infinity instance, create table, insert data, search the data
"""

# import infinity_embedded as infinity
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
    infinity_table = infinity_db.get_table("CQADupStack_en_Table")
    infinity_table.drop_index('CQADupStack_en_tensor_index',ConflictType.Ignore)
    print("Start creating tensor index...")
    begin_hnsw_time = time.time()
    res = infinity_table.create_index("CQADupStack_en_tensor_index", index.IndexInfo("tensor_col", index.IndexType.EMVB,
                                    {
                                        "pq_subspace_num":"32",
                                        "pq_subspace_bits":"8",
                                    }
                                ),
                            ConflictType.Error)
    end_hnsw_time = time.time()
    infinity_table.show_index("CQADupStack_en_tensor_index")
    print("end tensor time: ",(end_hnsw_time - begin_hnsw_time)*1000,'ms')
    print("Finish creating tensor index.")
    assert res.error_code == ErrorCode.OK

    print('build done')
    sys.exit(0)
except Exception as e:
    print(str(e))
    sys.exit(-1)

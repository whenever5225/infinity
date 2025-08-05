

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
    infinity_table.drop_index('mldr_en_hnsw_index',ConflictType.Ignore)
    print("Start creating Hnsw index...")
    begin_hnsw_time = time.time()
    res = infinity_table.create_index("mldr_en_hnsw_index", index.IndexInfo("dense_col", index.IndexType.Hnsw,
                                                                            {
                                                                                "m": "16",
                                                                                "ef_construction": "200",
                                                                                "metric": "ip",
                                                                                "encode": "lvq"
                                                                            }),
                                            ConflictType.Error)
    end_hnsw_time = time.time()
    print(infinity_table.show_index("mldr_en_hnsw_index"))
    print("end hnsw time: ",(end_hnsw_time - begin_hnsw_time)*1000,'ms')
    print("Finish creating Hnsw index.")
    assert res.error_code == ErrorCode.OK

    print('build done')
    sys.exit(0)
except Exception as e:
    print(str(e))
    sys.exit(-1)

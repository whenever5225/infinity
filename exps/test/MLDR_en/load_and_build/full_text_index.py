

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
    print("Start creating fulltext index.")
    if lang == "zh":
        ft_params = {"ANALYZER": "chinese"}
    
    infinity_db = infinity_instance.get_database('default_db')
    infinity_table = infinity_db.get_table("mldr_en_Table")
    infinity_table.drop_index('mldr_en_ft_index',ConflictType.Ignore)
    begin_full_text_time = time.time()
    res = infinity_table.create_index("mldr_en_ft_index",
                                            index.IndexInfo("fulltext_col", index.IndexType.FullText, ft_params),
                                            ConflictType.Error)
    infinity_table.show_index("mldr_en_ft_index")
    end_full_text_time = time.time()
    print(infinity_table.show_index('mldr_en_ft_index'))
    print("end full text time: ",(end_full_text_time - begin_full_text_time)*1000,'ms')
    assert res.error_code == ErrorCode.OK

    print('build done')
    sys.exit(0)
except Exception as e:
    print(str(e))
    sys.exit(-1)



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
    print("Start creating fulltext index.")
    if lang == "zh":
        ft_params = {"ANALYZER": "chinese"}
    
    hybridsearch_db = hybridsearch_instance.get_database('default_db')
    hybridsearch_table = hybridsearch_db.get_table("mldr_en_Table")
    hybridsearch_table.drop_index('mldr_en_ft_index',ConflictType.Ignore)
    begin_full_text_time = time.time()
    res = hybridsearch_table.create_index("mldr_en_ft_index",
                                            index.IndexInfo("fulltext_col", index.IndexType.FullText, ft_params),
                                            ConflictType.Error)
    hybridsearch_table.show_index("mldr_en_ft_index")
    end_full_text_time = time.time()
    print(hybridsearch_table.show_index('mldr_en_ft_index'))
    print("end full text time: ",(end_full_text_time - begin_full_text_time)*1000,'ms')
    assert res.error_code == ErrorCode.OK

    print('build done')
    sys.exit(0)
except Exception as e:
    print(str(e))
    sys.exit(-1)

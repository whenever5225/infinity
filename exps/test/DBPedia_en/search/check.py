import os
import time
from DBPedia_en.search.vec_read import load_sparse
from datasets import Dataset
import pandas as pd
import hybridsearch
from hybridsearch.common import LOCAL_HOST

#  CSV 
# df = pd.read_csv('data.csv')
# 
pd.options.display.max_columns = None
# 
pd.options.display.max_rows = None
# 
pd.options.display.max_colwidth = None

 #  Use hybridsearch module to connect a remote server
hybridsearch_instance = hybridsearch.connect(LOCAL_HOST)
# hybridsearch_instance.drop_database("default_db2")
# 'default_db' is the default database
db_instance = hybridsearch_instance.get_database("default_db")
table = db_instance.get_table('DBPedia_en_Table')
table.drop_index('DBPedia_en_ft_index')
print(db_instance.show_table('DBPedia_en_Table'))
# print(db_instance.drop_table('DBPedia_en_Table6'))
# corpus = pd.read_csv('/home/ubuntu/data_download_data/embedding_reserve/DBPedia_en/english/DBPedia_en_combine_corpus.csv')
# specific_id_list = ['102236','91901','177507','80798','112990','182056','130867']
# for specific_id in specific_id_list:
#     docid_list = corpus.query(f'_id == {specific_id}')
#     print(docid_list['combine_text_and_title'].to_string(index=False))

dense_embedding_dir = '/home/ubuntu/data_download_data/embedding_reserve/DBPedia_en/dense_embeddings/vectors'

sparse_embedding_dir = '/home/ubuntu/data_download_data/embedding_reserve/DBPedia_en/sparse_embeddings/vectors'
# sparse_names = [f for f in os.listdir(sparse_embedding_dir) if os.path.isfile(os.path.join(sparse_embedding_dir, f))]
# print("Start inserting data...")
# sparse_data = None
# sparse_file_idx = 0
# doc_id_idx = 0
# sparse_idx = 0
# sparse_nums = 0
# begin_insert_time = time.time()
# while sparse_file_idx < len(sparse_names):
#     sparse_path = os.path.join(sparse_embedding_dir,sparse_names[sparse_file_idx])
#     sparse_vectors = load_sparse(sparse_path)

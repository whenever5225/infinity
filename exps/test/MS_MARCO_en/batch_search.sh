# data insert
# python3 $PWD/build/data_insert.py

# index build
# python3 $PWD/build/dense_index.py
# python3 $PWD/build/sparse_index.py
# python3 $PWD/build/full_text_index.py

## rrf
### single road
# python3 $PWD/search/single_road/dense_search.py
# python3 $PWD/search/single_road/fulltext_search.py
# python3 $PWD/search/single_road/sparse_search.py
### two roads
# python3 $PWD/search/two_roads/dense_fulltext_search.py
python3 $PWD/search/two_roads/sparse_dense_search.py
python3 $PWD/search/two_roads/sparse_fulltext_search.py
### three roads
python3 $PWD/search/three_roads/dense_fulltext_sparse.py

python3 /home/ubuntu/hybridsearch/experiments/load_experiments/MLDR_zh/build/data_insert.py

# ## tensor
# ### two roads
# python3 $PWD/search_tensor_rank/two_roads/dense_fulltext_search.py
# python3 $PWD/search_tensor_rank/two_roads/sparse_dense_search.py
# python3 $PWD/search_tensor_rank/two_roads/sparse_fulltext_search.py
# ### three roads
# python3 $PWD/search_tensor_rank/three_roads/sparse_dense_fulltext_search.py

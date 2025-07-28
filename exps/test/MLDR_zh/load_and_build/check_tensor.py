import os
import re
import struct
import numpy as np
from tqdm import tqdm

def extract_number(filename):
    match = re.search(r'(\d+)\.', filename)
    if match:
        return int(match.group(1))
    return 0

def load_colbert_list(multivec_save_file: str):
    multivec_embeddings = []
    with open(multivec_save_file, 'rb') as f:
        while True:
            # l
            l_bytes = f.read(4)
            if not l_bytes:
                break  # 
            l = struct.unpack('<i', l_bytes)[0]

            # 
            one_multivec = []
            for _ in range(l):
                # dim
                dim = struct.unpack('<i', f.read(4))[0]

                #  dim  float16 
                vec = np.fromfile(f, dtype=np.float16, count=dim)
                one_multivec.append(vec)

            multivec_embeddings.append(np.array(one_multivec))

    return multivec_embeddings

tensor_embedding_dir = '/home/ubuntu/large_disk/hybridsearch/mldr_zh/tensor_embeddings/vectors'
tensor_names = [f for f in os.listdir(tensor_embedding_dir) if os.path.isfile(os.path.join(tensor_embedding_dir, f))]
tensor_names = sorted(tensor_names, key=extract_number)
# load_colbert_list("/home/ubuntu/large_disk/hybridsearch/mldr_zh/tensor_embeddings/vectors/.mldr_zh_bert265.fvecs.OMIqW3")
load_colbert_list("/home/ubuntu/large_disk/hybridsearch/mldr_zh/tensor_embeddings/vectors/mldr_zh_bert265.fvecs")
print(len(tensor_names))
tensor_names = tensor_names[250:]
tensor_file_idx = 0
print(len(tensor_names))
while tensor_file_idx < len(tensor_names):
    tensor_path = os.path.join(tensor_embedding_dir,tensor_names[tensor_file_idx])
    tensor_vectors = load_colbert_list(tensor_path)
    print(tensor_path)
    tensor_file_idx += 1
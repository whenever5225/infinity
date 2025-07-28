import struct
import numpy as np
from tqdm import tqdm

def load_colbert_list(multivec_save_file: str):
    print(multivec_save_file)
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

# # 
# multivec_save_file = '/home/ubuntu/data_download_data/embedding_reserve/Multilingual_CC_News_zh/tensor_embeddings/vectors/Multilingual_CC_News_zh_bert1.fvecs'
# multivec_embeddings = load_colbert_list(multivec_save_file)

# # 
# print(f"Loaded {len(multivec_embeddings)} embeddings.")
# for i, multivec in enumerate(multivec_embeddings[:3]):  # 3
#     print(f"Embedding {i}: shape = {multivec.shape}")

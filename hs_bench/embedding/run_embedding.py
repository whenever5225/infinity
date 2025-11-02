import argparse
import json
import struct
from pathlib import Path
import sys
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass, field
from transformers import HfArgumentParser

# Add project root to path to allow importing local libraries
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# Use mock generation if torch/transformers are not installed
try:
    import torch
    from FlagEmbedding import FlagModel, BGEM3FlagModel
    LIBS_AVAILABLE = True
except ImportError:
    LIBS_AVAILABLE = False
    print("Warning: `torch` or `FlagEmbedding` not found. Using mock embedding generation.")

@dataclass
class ModelArgs:
    encoder: str = field(default="BAAI/bge-m3", metadata={'help': 'Name or path of encoder'})
    fp16: bool = field(default=True, metadata={'help': 'Use fp16 in inference?'})
    pooling_method: str = field(default='cls', metadata={'help': "Pooling method. Avaliable methods: 'cls', 'mean'"})
    normalize_embeddings: bool = field(default=True, metadata={'help': "Normalize embeddings or not"})

@dataclass
class DataArgs:
    config: str = field(metadata={'help': 'Dataset config name (e.g., mldr_en)'})
    model_name: str = field(metadata={'help': 'Name of the embedding model to use (must be in config)'})
    target: str = field(default="corpus", metadata={'help': 'Target to generate embeddings for (corpus or queries)'})
    batch_size: int = field(default=32, metadata={'help': 'Batch size for embedding generation'})

def get_dense_model(model_args: ModelArgs):
    if not LIBS_AVAILABLE:
        return None
    return FlagModel(model_args.encoder,
                     pooling_method=model_args.pooling_method,
                     normalize_embeddings=model_args.normalize_embeddings,
                     use_fp16=model_args.fp16)

def get_sparse_model(model_args: ModelArgs):
    if not LIBS_AVAILABLE:
        return None
    return BGEM3FlagModel(model_args.encoder,
                          use_fp16=model_args.fp16)

def get_colbert_model(model_args: ModelArgs):
    if not LIBS_AVAILABLE:
        return None
    return FlagModel(model_args.encoder,
                     use_fp16=model_args.fp16)

def generate_dense(model, texts, batch_size):
    if not LIBS_AVAILABLE:
        print("Generating mock dense embeddings.")
        return np.random.rand(len(texts), 768).astype(np.float32)
    return model.encode(texts, batch_size=batch_size, max_length=max_length).astype(np.float32)

def generate_sparse(model, texts, batch_size, max_length=512):
    if not LIBS_AVAILABLE:
        print("Generating mock sparse embeddings.")
        return [{"indices": [i], "values": [np.random.rand()]} for i in range(len(texts))]
    result_dict = model.encode(texts, batch_size=batch_size, max_length=max_length, return_dense=False, return_sparse=True, return_colbert_vecs=False)
    sparse_dict = result_dict['lexical_weights']
    result = []
    for one_dict in sparse_dict:
        one_result = {}
        for p, v in one_dict.items():
            one_result[p] = float(v)
        result.append(one_result)
    return result

def generate_colbert(model, texts, batch_size, max_length=512):
    if not LIBS_AVAILABLE:
        print("Generating mock colbert embeddings.")
        return [np.random.rand(10, 128).astype(np.float32) for _ in range(len(texts))]
    result_dict = model.encode(texts, batch_size=batch_size, max_length=max_length, return_dense=False, return_sparse=False, return_colbert_vecs=True)
    return result_dict['colbert_vecs']

def save_dense_embeddings(embeddings, output_file):
    print(f"Saving dense embeddings to: {output_file}")
    np.save(output_file, embeddings)

def save_sparse_embeddings(embeddings, output_file):
    print(f"Saving sparse embeddings to: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(embeddings, f)

def save_colbert_embeddings(embeddings, output_file):
    print(f"Saving Colbert embeddings to: {output_file}")
    with open(output_file, 'wb') as f:
        for one_multivec in tqdm(embeddings, desc="Saving multivec embeddings"):
            l, dim = one_multivec.shape
            f.write(struct.pack('<i', l))
            for vec in one_multivec:
                f.write(struct.pack('<i', dim))
                vec.astype('float32').tofile(f)

def main():
    parser = HfArgumentParser((DataArgs,))
    data_args, = parser.parse_args_into_dataclasses()

    config_path = project_root / "hs_bench" / "configs" / f"{data_args.config}.json"
    print(f"Loading config from: {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Find model info in config
    model_info = next((m for m in config["embedding_models"] if m["name"] == data_args.model_name), None)
    if not model_info:
        raise ValueError(f"Model '{data_args.model_name}' not found in config file {config_path}")

    # Create ModelArgs from config
    params = model_info.get("params", {})

    # The 'encoder' or 'model_name' in 'params' specifies the actual model path for the library
    encoder_path = params.get("encoder") or params.get("model_name")
    if not encoder_path:
        raise ValueError(f"Config for model '{data_args.model_name}' is missing 'encoder' or 'model_name' in 'params'.")

    model_args = ModelArgs(
        encoder=encoder_path,
        fp16=params.get("fp16", True),
        pooling_method=params.get("pooling_method", 'cls'),
        normalize_embeddings=params.get("normalize_embeddings", True)
    )
    embedding_type = model_info.get("storage_type", "dense")


    # Determine input file and output directory
    if data_args.target == "corpus":
        input_path = project_root / config['dataset']['corpus_path']
        output_dir = project_root / config['dataset']['corpus_embeddings_dir']
    else:  # queries
        input_path = project_root / config['dataset']['queries_path']
        output_dir = project_root / config['dataset']['queries_embeddings_dir']
    
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load texts to embed
    print(f"Loading {data_args.target} from: {input_path}")
    texts_to_embed = []
    if data_args.target == "corpus":
        with open(input_path, 'r') as f:
            for line in f:
                doc = json.loads(line)
                texts_to_embed.append(doc.get('title', '') + " " + doc.get('text', ''))
    else: # queries are in tsv format: qid\ttext
        with open(input_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    texts_to_embed.append(parts[1])

    # Generate Embeddings
    if embedding_type == "dense":
        model = get_dense_model(model_args)
        embeddings = generate_dense(model, texts_to_embed, data_args.batch_size)
        output_path = output_dir / f"{data_args.model_name}.npy"
        save_dense_embeddings(embeddings, output_path)
    elif embedding_type == "sparse":
        model = get_sparse_model(model_args)
        embeddings = generate_sparse(model, texts_to_embed, data_args.batch_size)
        output_path = output_dir / f"{data_args.model_name}.json"
        save_sparse_embeddings(embeddings, output_path)
    elif embedding_type == "tensor":
        model = get_colbert_model(model_args)
        embeddings = generate_colbert(model, texts_to_embed, data_args.batch_size)
        output_path = output_dir / f"{data_args.model_name}.data"
        save_colbert_embeddings(embeddings, output_path)
    else:
        raise ValueError(f"Unsupported embedding type in config: {embedding_type}")

    print("Embedding generation completed.")

if __name__ == "__main__":
    main()
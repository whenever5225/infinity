# Balancing the Blend: An Experimental Analysis of Trade-offs in Hybrid Search

*Official Implementation for our paper submitted to VLDB 2026*

[**Paper PDF**](paper.pdf) | [**Project Website**](https://github.com/infiniflow/infinity) | [**License: Apache 2.0**](#license)

---

## 📖 Overview

This repository contains the full implementation and experimental data for our paper, **"Balancing the Blend: An Experimental Analysis of Trade-offs in Hybrid Search".**

Hybrid search, combining lexical and semantic retrieval, is now a foundational technology for modern information retrieval. However, the architectural design space for these systems is vast and complex, yet a systematic, empirical understanding of the trade-offs among their core components—retrieval paradigms, combination schemes, and re-ranking methods—is critically lacking.

This work presents the **first systematic benchmark** of advanced hybrid search architectures, informed by our experience building the [**Infinity**](https://github.com/infiniflow/infinity) open-source database. We evaluate four retrieval paradigms (FTS, SVS, DVS, TenS) and their 15 combinations across 11 real-world datasets to provide a data-driven map of the performance landscape. This repository provides all the necessary tools to reproduce our findings and to extend this research.

## 🚀 Getting Started

### Prerequisites
- CPU: x86_64 with AVX2 support.
- OS:
  - Linux with glibc 2.17+.
  - Windows 10+ with WSL/WSL2.
  - MacOS
- Python: Python 3.10+.

### Install Infinity server

#### Linux x86_64 & MacOS x86_64

```bash
sudo mkdir -p /var/infinity && sudo chown -R $USER /var/infinity
docker pull infiniflow/infinity:nightly
docker run -d --name infinity -v /var/infinity/:/var/infinity --ulimit nofile=500000:500000 --network=host infiniflow/infinity:nightly
```
#### Windows

If you are on Windows 10+, you must enable WSL or WSL2 to deploy Infinity using Docker. Suppose you've installed Ubuntu in WSL2:

1. Follow [this](https://learn.microsoft.com/en-us/windows/wsl/systemd) to enable systemd inside WSL2.
2. Install docker-ce according to the [instructions here](https://docs.docker.com/engine/install/ubuntu).
3. If you have installed Docker Desktop version 4.29+ for Windows: **Settings** **>** **Features in development**, then select **Enable host networking**.
4. Pull the Docker image and start Infinity: 

   ```bash
   sudo mkdir -p /var/infinity && sudo chown -R $USER /var/infinity
   docker pull infiniflow/infinity:nightly
   docker run -d --name infinity -v /var/infinity/:/var/infinity --ulimit nofile=500000:500000 --network=host infiniflow/infinity:nightly
   ```

### Install Infinity client

```
pip install infinity-sdk==0.6.0.dev4
```

### 🔧 Deploy Infinity using binary

If you wish to deploy Infinity using binary with the server and client as separate processes, see the [Deploy infinity using binary](https://infiniflow.org/docs/dev/deploy_infinity_server) guide.

### 🔧 Build from Source

See the [Build from Source](https://infiniflow.org/docs/dev/build_from_source) guide.

## 📊 Running the Benchmark

All scripts and results are located in the `exps/` directory. For detailed documentation on the underlying Python API, please refer to the [official documentation](https://infiniflow.org/docs/dev/pysdk_api_reference).

### Step 1: Download and Prepare Datasets
We provide direct download links for all 11 datasets in the table below (see [Datasets](#-datasets)). Please download the datasets you wish to evaluate and place the unzipped contents into the corresponding subdirectories within the `exps/test/` directory.

To generate the dense vectors, sparse vectors, and tensors for a given dataset, please refer to the scripts provided in the `python/benchmark/mldr_benchmark/` directory.

### Step 2: Import Data
For each dataset, run the `data_insert.py` script located in its subdirectory. This will load the texts, dense vectors, sparse vectors, and tensors into the hybrid search system.

### Step 3: Build Indexes
After importing the data, run the following scripts in the same directory to build the specialized index for each retrieval paradigm:
- `fulltext_index.py`: Constructs the full-text index for FTS.
- `sparse_index.py`: Constructs the sparse vector index for SVS.
- `dense_index.py`: Constructs the dense vector index for DVS.
- `tensor_index.py`: Constructs the tensor index for TenS.

### Step 4: Run Retrieval Experiments
You can run experiments for specific types of architectures. The scripts are organized into the following directories within `exps/test/[dataset]/search/`:

- `single_road/`: Contains scripts to evaluate the performance of the four individual retrieval paradigms (FTS, SVS, DVS, TenS).
- `two_road/`: Contains scripts to test all six two-path retrieval combinations (e.g., FTS + DVS).
- `three_road/`: Contains scripts to assess all four three-path retrieval combinations (e.g., FTS + DVS + SVS).
- `four_road/`: Contains the script to evaluate the performance of the combined four-path approach.

> **Note:** To run all retrieval combinations on a single dataset, navigate to the corresponding experiment directory and execute its `batch_search.sh` script. For example, to run all experiments for the `CQAD(en)` dataset:
```bash
cd exps/test/CQADupStack_en
bash batch_search.sh
```

## 📈 Results

A comprehensive analysis of the experimental results is presented in the main paper. For additional details, including a full breakdown of the results, please refer to the supplementary materials available [here](exps/results/README.md).

## 📚 Datasets

Our evaluation uses 11 diverse real-world datasets. The table below provides statistics and download links.

| Dataset  | Domain           | Task                           | #Corpus   | #Query | Download Link |
| :------- | :--------------- | :----------------------------- | :-------- | :----- | :--- |
| MSMA(en) | Miscellaneous    | Passage Retrieval              | 8,841,823 | 43     | [Link](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/msmarco.zip) |
| DBPE(en) | Wikipedia        | Entity Retrieval               | 4,635,922 | 400    | [Link](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/dbpedia-entity.zip) |
| MCCN(zh) | News             | Question Answering             | 935,162   | 339    | [Link](https://huggingface.co/datasets/intfloat/multilingual_cc_news) |
| TOUC(en) | Miscellaneous    | Argument Retrieval             | 382,545   | 49     | [Link](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/webis-touche2020.zip) |
| MLDR(zh) | Wikipedia, Wudao | Long-Document Retrieval        | 200,000   | 800    | [Link](https://huggingface.co/datasets/Shitao/MLDR) |
| MLDR(en) | Wikipedia        | Long-Document Retrieval        | 200,000   | 800    | [Link](https://huggingface.co/datasets/Shitao/MLDR) |
| TREC(en) | Bio-Medical      | Bio-Medical Information Retrieval | 171,332   | 50     | [Link](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/trec-covid.zip) |
| FIQA(en) | Finance          | Question Answering             | 57,638    | 648    | [Link](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip) |
| CQAD(en) | StackExchange    | Duplicate-Question Retrieval   | 40,221    | 1,570  | [Link](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/cqadupstack.zip) |
| SCID(en) | Scientific       | Citation Prediction            | 25,657    | 1,000  | [Link](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scidocs.zip) |
| SCIF(en) | Scientific       | Fact Checking                  | 5,183     | 809    | [Link](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip) |

*For detailed statistics including corpus sizes, please refer to the `README.md` file in the `exps/datasets/` directory.*

## 📜 Citation

If you find this work useful for your research, please consider citing our paper:

```bibtex
@article{hybridsearch25-infinity,
  title={Balancing the Blend: An Experimental Analysis of Trade-offs in Hybrid Search},
  author={Wang, Mengzhao and Tan, Boyu and Gao, Yunjun and Jin, Hai and Zhang, Yingfeng and Ke, Xiangyu and Xu, Xiangliang and Zhu, Yifan},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2025}
}
```

## 🤝 Contribution
We welcome contributions from the community to improve and extend this benchmark framework. We encourage researchers and developers to help by reporting bugs, proposing new features by opening an issue, or submitting code changes via a pull request. To ensure a smooth collaboration, please first refer to our detailed contribution guidelines outlined in [CONTRIBUTING.md](./CONTRIBUTING.md).

## 📄 License

This project is licensed under the **Apache 2.0 License**. See the [LICENSE](./LICENSE) file for details.

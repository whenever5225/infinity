# Balancing the Blend: An Experimental Analysis of Trade-offs in Hybrid Search

*Official Artifact for the paper "Balancing the Blend: An Experimental Analysis of Trade-offs in Hybrid Search" submitted to VLDB 2026*

[**Paper PDF**](https://arxiv.org/pdf/2508.01405) | [**Project Website**](https://github.com/infiniflow/infinity) | [**License: Apache 2.0**](#license)

---

## 📖 Overview

Hybrid search, combining lexical and semantic retrieval, is now a foundational technology for modern information retrieval. However, the architectural design space for these systems is vast and complex, yet a systematic, empirical understanding of the trade-offs among their core components—retrieval paradigms, combination schemes, and re-ranking methods—is critically lacking.

This work presents the **first systematic benchmark** of advanced hybrid search architectures, informed by our experience building the [**Infinity**](https://github.com/infiniflow/infinity) open-source database. We evaluate four retrieval paradigms (FTS, SVS, DVS, TenS) and their 15 combinations across 11 real-world datasets to provide a data-driven map of the performance landscape. This repository provides all the necessary tools to reproduce our findings and to extend this research.

## 🚀 Getting Started

- **Infinity**: Please follow the instructions on the [Project Website](https://github.com/infiniflow/infinity) to install the Infinity database.
- **Python Dependencies**:
  ```shell
  pip install -r requirements.txt
  ```

## 🏛️ Repository Structure

This repository is split into two main components to serve different purposes:

1.  **`exps/`**: **Paper Reproducibility**
    This directory provides a set of scripts to reproduce all experimental results reported in our paper. If your goal is to validate and reproduce our findings, this is the place to start. See `exps/README.md` for instructions.

2.  **`hs_bench/`**: **Evaluation Framework**
    This directory contains the modular scripts for the evaluation pipeline (configs, embedding, import, indexing, retrieval, and evaluation). This is the code to modify if you want to extend our work—for example, by adding new datasets or embedding models. See `hs_bench/README.md` for the full technical guide.

### How to Use

**To Reproduce Paper Results:**

1.  Clone this repository.
2.  Navigate to `exps/` and follow the `exps/README.md` instructions.
3.  Download datasets as per `exps/datasets/README.md`.
4.  Run the scripts in `exps/scripts/` to build indexes, run all experiments, and evaluate the results.

**To Extend This Work:**

1.  Clone this repository.
2.  Navigate to `hs_bench/` and read the `hs_bench/README.md`.
3.  Follow the guide to add your own configs, models, or datasets.

## 📜 Citation

If you find this work useful for your research, please consider citing our paper:

```bibtex
@article{hybridsearch25-infinity,
  title={Balancing the Blend: An Experimental Analysis of Trade-offs in Hybrid Search},
  author={Wang, Mengzhao and Tan, Boyu and Gao, Yunjun and Jin, Hai and Zhang, Yingfeng and Ke, Xiangyu and Xu, Xiangliang and Zhu, Yifan},
  journal={arXiv preprint arXiv:2508.01405},
  year={2025}
}
```

## 🤝 Contribution

We welcome contributions from the community to improve and extend this benchmark framework. We encourage researchers and developers to help by reporting bugs, proposing new features by opening an issue, or submitting code changes via a pull request. To ensure a smooth collaboration, please first refer to our detailed contribution guidelines outlined in [CONTRIBUTING.md](./CONTRIBUTING.md).

## 📄 License

This project is licensed under the **Apache 2.0 License**. See the [LICENSE](./LICENSE) file for details.

## 🙏 Acknowledgements

The design and implementation of the evaluation framework in this repository were significantly informed by our work on the [Infinity](https://github.com/infiniflow/infinity) open-source database. We are grateful to our colleagues on the Infinity team for their foundational work and insightful discussions. A special thanks to **Zhichang Yu**, **Yushi Shen**, **Zhiqiang Yang**, **Ling Qin**, and **Yi Xiao** for their invaluable contributions to the Infinity project.
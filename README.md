# Little Giants: Synthesizing High-Quality Embedding Data at Scale


<p align="center">
📖 <a href="https://arxiv.org/pdf/2410.18634" target="_blank">Paper</a> • 🤗 <a href="https://huggingface.co/Haon-Chen/speed-synthesis-7b-senior" target="_blank">Senior Generator</a> • <a href="https://huggingface.co/Haon-Chen/speed-synthesis-7b-revisor" target="_blank">Data Revisor</a> • <a href="https://huggingface.co/Haon-Chen/speed-embedding-7b-instruct" target="_blank">Embedding Model</a>  <br>
</p>

---

## Usage

Use the [classification tryout](https://github.com/haon-chen/SPEED/blob/main/senior_model_tryout.py) we provide and the above [generators](https://huggingface.co/Haon-Chen/speed-synthesis-7b-senior) to synthesize your own embedding data!

Use our [embedding model](https://huggingface.co/Haon-Chen/speed-embedding-7b-instruct) to perform all kinds of embedding tasks!

## Abstract

<img width="665" alt="introduction" src="https://github.com/user-attachments/assets/2d0c34f3-07d1-4d49-bacf-af19fd9722be">

Synthetic data generation has become an increasingly popular way of training models without the need for large, manually labeled datasets. For tasks like text embedding, synthetic data offers diverse and scalable training examples, significantly reducing the cost of human annotation. However, most current approaches rely heavily on proprietary models like GPT-4, which are expensive and inefficient for generating large-scale embedding data. In this paper, we introduce **SPEED**, a framework that aligns open-source small models (8B) to efficiently generate large-scale synthetic embedding data. Through supervised fine-tuning, preference optimization, and self-improvement, **SPEED** enables small open-source models to produce high-quality data.  Remarkably, **SPEED** uses only less than 1/10 of the GPT API calls, outperforming the state-of-the-art embedding model E5$_\text{mistral}$ when both are trained solely on their synthetic data. Using this efficient generator, we conduct a comprehensive study on how various factors within the alignment pipeline impact data quality and reveal the scaling law for synthetic embedding data.

<img width="1353" alt="framework" src="https://github.com/user-attachments/assets/a0a49635-5a0b-4c1e-9e5a-504e512e7cd2">

## Citation
Please kindly cite our paper if helps your research:
```BibTex
@article{chen2024little,
  title={Little Giants: Synthesizing High-Quality Embedding Data at Scale},
  author={Chen, Haonan and Wang, Liang and Yang, Nan and Zhu, Yutao and Zhao, Ziliang and Wei, Furu and Dou, Zhicheng},
  journal={arXiv preprint arXiv:2410.18634},
  year={2024}
}
```

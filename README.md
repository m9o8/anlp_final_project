<h1 align="center">Patent Classification with Transformers</h1>
<p align="center">
  <b>Advanced NLP - Final Project</b><br>
  <b>Ferran Boada Bergadá, Lucia Sauer, Julian Romero & Moritz Peist</b><br>
  Barcelona School of Economics · 2025 
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10-blue?logo=python">

  <img src="https://img.shields.io/badge/BERT-NLP%20Model-9cf?logo=google&logoColor=white" />
  <img src="https://img.shields.io/badge/Transformers-HuggingFace-yellow?logo=huggingface&logoColor=black" />

</p>

---



## 🧠 Project Overview

This project focuses on **classifying patent documents** into 9 categories based on the **Cooperative Patent Classification (CPC)** scheme. We leverage the [`ccdv/patent-classification`](https://huggingface.co/datasets/ccdv/patent-classification) dataset from Hugging Face, which originates from **Google Patents Public Datasets via BigQuery** (Google, 2018).

Originally designed for **abstractive summarization**, this dataset is repurposed here for **multi-class classification**. It consists of:
- **25,000 training samples**
- **5,000 samples for validation and testing**
- Full patent texts and associated 9 CPC labels

### 📊 Dataset Distribution

The dataset shows strong class imbalance, with most patents concentrated in a few categories—especially from fast-evolving fields like electronics and healthcare. This is common in patent corpora and follows a Pareto-like distribution, as noted by [Benzineb & Guyot (2011)](#ref1).

To fairly evaluate performance, we use **Macro F1** that treats all classes equally, highlighting minority performance and **Micro F1** that favors frequent classes by aggregating across all samples.

These metrics give a more balanced view than accuracy alone.

<div align="center">
<img src="results/plots/class_imbalance.png" alt="Text Length Distribution" 
width="80%"/>
</div>
Patent documents are inherently long and verbose, often containing multiple sections like the title, abstract, description, and claims. As shown below, their lengths vary widely—with a heavily **right-skewed** distribution across all classes, and some texts reaching up to 80,000 tokens.

This extreme variability poses computational and modeling challenges, especially for transformer models limited to shorter input lengths.

<div align="center">
  <img src="results/plots/text_lenght_distribution.png" alt="Text Length Distribution" width="80%"/>
</div>


To explore vocabulary differences across categories, the following word clouds visualize the most frequent terms per class. While **domain-specific terms** clearly emerge, many words are **shared across categories**, introducing ambiguity and making it harder for models to distinguish between classes. Additionally, patents often contain **dense scientific language** and are **intentionally phrased in complex or obfuscated ways**—either to avoid plagiarism or obscure related inventions. These characteristics amplify the challenge of accurate classification, as key signals may be both rare and linguistically subtle.




<div align="center">
  <img src="results/plots/wordcloud.png" alt="Text Length Distribution" width="80%"/>
</div>

---

## 🎯 Objective, Challenges & Relevance

The goal of this project is to **automate the classification of patents** into CPC categories using modern Transformer-based models. Accurate classification is critical for routing patents to the right domain experts, accelerating the innovation pipeline and reducing administrative burden.

However, the task comes with significant challenges:
- The dataset is **highly imbalanced**, reflecting real-world skew in patent filings.
- **Patent texts are long**, often surpassing the 512-token limit of base Transformer models.
- **Technical vocabulary** and subtle distinctions between classes add further complexity.

Despite these challenges, **automating patent classification** has important real-world impact. It helps address the increasing volume and complexity of patent filings, reduces manual workload for experts, and ensures more consistent and scalable classification—ultimately speeding up how innovation is reviewed and protected.

---


## 📈 Related Work & SOA Benchmarks

Below is a comparison of recent studies tackling the patent classification task using machine learning and Transformer-based architectures:

| Model / Method                     | Dataset Used                                   | #Classes | Performance                  | Reference |
|----------------------------------|--------------------------------------------------|----------|------------------------------|-----------|
| RoBERTa (512 tokens)             | ccdv/patent-classification                       | CPC  (9) | 66.6 / 61.8  (Micro / Macro) | [Condevaux & Harispe (2023)](#ref2) |
| LSG-Norm Attention (128/4)       | ccdv/patent-classification                       | CPC  (9) | 70.0 / 64.4  (Micro / Macro) | [Condevaux & Harispe (2023)](#ref3) |
| PatentBERT                       | USPTO-3M (claims only)                           | CPC  (9) | 66.80   (F1)                 | [Lee & Hsiang (2020)](#ref2) |
| Optimized Neural Networks (MLP)  | WIPO-alpha (English patents)                     | CPC  (9) | —      (Accuracy)            | [Abdelgawad et al. (2022)](#ref4) |

---

## 📚 References
<a id="ref1"></a>**[1]** Benzineb & Guyot (2011). *Automated Patent Classification*.  
<a id="ref2"></a>**[1]** Condevaux, C. & Harispe, S. (2023). *LSG Attention: Extrapolation of pretrained Transformers to long sequences*.  
<a id="ref3"></a>**[2]** Lee, J.-S., & Hsiang, J. (2020). *Patent classification by fine-tuning BERT language model*.  
<a id="ref4"></a>**[3]** Abdelgawad, L., Kluegl, P., Genc, E., Falkner, S., & Hutter, F. (2022). *Optimizing Neural Networks for Patent Classification*.  

---

## ⚙️ Project Structure

```bash
anlp_final_project/
├── code/                      # Jupyter Notebooks for each exercise
│   ├── 📓 task1.ipynb
|   |── 📓 task2.ipynb
|   |── 📓 task3.ipynb
|   |── 📓 task4.ipynb
├── src/                            # Python helper function
│   ├── utils.py
├── results/                        # Plots and model evaluations
│   ├── plots/
├── uv.lock
├── pyproject.toml
├── .python-version
└── README.md
````
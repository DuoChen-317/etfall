# 🧠 Emotional Bias Detection and Mitigation in LLMs

This project investigates **emotional bias** in large language models (LLMs) across multiple languages.  
It evaluates LLM outputs’ **emotional tone and toxicity** using the [Detoxify](https://github.com/unitaryai/detoxify) model and explores possible **bias mitigation strategies**.

---


## 📖 Table of Contents

- [🚀 Overview](#️-Overview)  
- [📊 Project Status](#-Project-Status)  
- [📦 Environment Setup](#-eEnvironment-Setup)  
- [📄 License](#-license)  

---


## 🚀 Overview

| Component | Description |
|------------|-------------|
| **Goal** | Detect and analyze emotional bias in LLM-generated text accrossing different languages |
| **Models** | Qwen2.5, LLaMA-3, or other open LLMs (via vLLM) |
| **Evaluator** | Detoxify (for toxicity & emotion scoring) |
| **Dataset** | [XNLI](https://huggingface.co/datasets/facebook/xnli) multilingual benchmark |
| **Frameworks** | `vLLM`, `datasets`|
| **Environment** | Singularity container with GPU support |

---

## 📊 Project Status

| Phase | Description | Status | Notes |
|:------|:-------------|:--------|:------|
| **1️⃣ Demo 1** | Environment setup and benchmark completed | ✅ Completed | `benchmark.py` |
| **2️⃣ Demo 2** | Experiment configuration and environment replication | 🔵 Planned | `evl_[model_name].py` |
| **3️⃣ Demo 3** | Try some possible way to mitigate the bias| 🔵 Planned | prompts etc. |
| **4️⃣ Report** | Analysis, visualization, and report writing | 🔵 Planned | Summarize results, discuss and reporting |

---


## 📦 Environment Setup

### Pull or Build the Singularity Image
This container includes the **vLLM framework** for LLM deployment, along with essential dependencies such as `datasets`, `pandas`, and `torch`.

```bash
singularity pull vllm_base.sif docker://tiyamo/vllm_base
```

### Install Packages for Emotional Detection
Install Detoxify
, a transformer-based model used for emotional and toxicity analysis

```bash
pip install detoxify
```

--- 


## 📄 License
This project is MIT Licensed. See `LICENSE` for details.


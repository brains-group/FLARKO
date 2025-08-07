# FLARKO: Financial Language-model for Asset Recommendation with Knowledge-graph Optimization

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Flower](https://img.shields.io/badge/Flower-1.19.0-green.svg)](https://flower.dev/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

FLARKO is a framework specifically designed for financial asset recommendation systems, combining knowledge graphs and large language models. FLARKO comes in two flavors: CenFLARKO (operates FLARKO in a centralized setting), and FedFLARKO (operates FLARKO in a federated setting). 

## Key Features

- **Knowledge Graph Optimization**: Graph-based data representation for financial assets
- **LLM Support**: Qwen3, Gemma3, and other transformer models with LoRA Fine-tuning
- **Temporal Awareness**: Time-aware data filtering for realistic financial modeling
- **Federated Learning**: Distributed training using the Flower framework

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [License](#license)

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/your-username/FLARKO.git
cd FLARKO
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Data Preparation

Download the [Far-Trans dataset](https://researchdata.gla.ac.uk/1658/) and place the CSV files in `./data/FAR-Trans`.

```bash
# Create financial datasets
cd data
python data/createFinData.py
# IID data
python data/createFinDataNorm.py
```

### 2. Centralized Training

```bash
# Start centralized training
python centralized_train.py --cfg_file centralized_full
```

### 3. Federated Training

```bash
# Start federated training
python train.py --cfg_file federated_full --num_rounds 200
```

### 4. Model Evaluation

```bash
# Evaluate trained models
python test.py --base_model_path Qwen/Qwen3-4B --lora_path ./models/path/to/lora/checkpoint
```

## Data Preparation

### Financial Dataset Structure

The system uses two main knowledge graphs:

1. **Background Knowledge Graph (non-behavioral knowledge)**: Market data, asset information, historical prices
2. **Transaction Knowledge Graphs (behavioral knowledge)**: Per-user transaction histories


## Training

### LLM Configuration

The chosen LLM can be trained normally by running `centralized_train.py`. The dataset, model, and hyperparameters can be adjusted by editing `centralized_full.yaml` in the `conf` folder.
Additionally, you can override the base model path by passing a `--base_model_path` argument. You can also override the dataset name used by passing the `--dataset_name` argument.

The chosen LLM can be trained via a federated simulation by running `train.py`. The dataset, model, and hyperparameters can be adjusted by editing `federated_full.yaml` in the `conf` folder.
Additionally, you can override the base model path or lora path by passing the `--base_model_path` or `--lora_path` arguments, respectively, and you can override the number of federated training rounds by passing the `--num_rounds` argument (useful for resuming from a checkpoint). You can also override the dataset name used by passing the `--dataset_name` argument.

Available configurations in `conf/`:

- **Federated**: `federated_full.yaml`, `federated_full_4B.yaml`, etc.
- **Centralized**: `centralized_full.yaml`, `centralized_full_8B.yaml`, etc.
- **Model Variants**: Qwen3 (0.6B, 4B, 8B), Gemma3 (1B, 4B)

### Custom Configuration

```yaml
# Create custom config
dataset:
  path: "./data/{}.json"
  name: "your_dataset"

model:
  name: "your_model"
  quantization: 4
  lora:
    peft_lora_r: 16
    peft_lora_alpha: 64

train:
  seq_length: 131072
  learning_rate: 5e-6

flower:
  num_rounds: 100
  sample_clients: 3
```


### Centralized Training

```bash
# Centralized training
python centralized_train.py --cfg_file centralized_full

# With custom model
python centralized_train.py \
  --cfg_file centralized_full_8B \
  --base_model_path Qwen/Qwen3-8B
```

### Federated Training

```bash
# Basic federated training
python train.py --cfg_file federated_full

# Custom configuration
python train.py \
  --cfg_file federated_full_4B \
  --base_model_path Qwen/Qwen3-4B \
  --num_rounds 100 \
  --dataset_name finDataset
```

Use `finDatasetNorm.json` instead of `findDataset.json` for the IID clients.

### Federated Training Configuration

Key parameters in configuration files:

```yaml
model:
  name: "Qwen/Qwen3-0.6B"
  quantization: 4  # 4-bit quantization
  lora:
    peft_lora_r: 16
    peft_lora_alpha: 64
    target_modules: ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

train:
  seq_length: 131072
  per_device_train_batch_size: 16
  learning_rate: 5e-6

flower:
  num_rounds: 200
  sample_clients: 3
  client_resources:
    num_cpus: 8
    num_gpus: 1.0
```

## Evaluation

### Metrics

Performance is measured using **Hits@3**, with three evaluation variants:

- **Pref@3:** Hit rate against *purchased assets*
- **Prof@3:** Hit rate against *profitable assets*
- **Comb@3:** Hit rate against assets that are both purchased and profitable (*desirable assets*)

### Evaluation Scripts

```bash
# Test model performance
python test.py --base_model_path Qwen/Qwen3-4B --lora_path ./models/path/to/lora/checkpoint
```

### Evaluation Output

**Performance of CenFLARKO across different model sizes and input configurations**
*Results are presented as mean ± standard error of a proportion. The best results for each model are in **bold**, and the best overall are marked with an ↑.*

|          Model | Data     |        Pref\@3       |        Prof\@3       |        Comb\@3       |
| -------------: | :------- | :------------------: | :------------------: | :------------------: |
| **Qwen3-0.6B** | Combined |    0.0354 ± 0.0131   |    0.1010 ± 0.0214   |    0.0152 ± 0.0087   |
|                | PKG      |  **0.4439 ± 0.0355** |    0.4694 ± 0.0356   |  **0.2551 ± 0.0311** |
|                | MKG      |    0.4141 ± 0.0350   |  **0.4747 ± 0.0355** |    0.2323 ± 0.0300   |
|                | Nothing  |    0.4352 ± 0.0357   |    0.4219 ± 0.0356   |    0.2240 ± 0.0301   |
| **Qwen3-1.7B** | Combined |    0.0990 ± 0.0216   |    0.4975 ± 0.0354   |    0.0524 ± 0.0161   |
|                | PKG      |    0.4434 ± 0.0483   |    0.4528 ± 0.0483   |    0.2642 ± 0.0428   |
|                | MKG      | **0.5341 ± 0.0532**↑ |    0.5169 ± 0.0530   |    0.3448 ± 0.0510   |
|                | Nothing  |    0.5000 ± 0.0566   |  **0.6154 ± 0.0551** | **0.3718 ± 0.0547**↑ |
|   **Qwen3-4B** | Combined |    0.2740 ± 0.0522   | **0.6400 ± 0.0554**↑ |    0.1644 ± 0.0434   |
|                | PKG      |  **0.2973 ± 0.0751** |    0.4324 ± 0.0814   |  **0.2973 ± 0.0751** |
|                | MKG      |    0.1250 ± 0.0523   |    0.1500 ± 0.0565   |    0.0750 ± 0.0416   |
|                | Nothing  |    0.1795 ± 0.0615   |    0.1316 ± 0.0548   |    0.1316 ± 0.0548   |
|   **Qwen3-8B** | Combined |    0.1136 ± 0.0478   |  **0.5909 ± 0.0741** |    0.0682 ± 0.0380   |
|                | PKG      |  **0.4528 ± 0.0684** |    0.5849 ± 0.0677   |  **0.3585 ± 0.0659** |
|                | MKG      |    0.2927 ± 0.0711   |    0.3415 ± 0.0741   |    0.2439 ± 0.0671   |
|                | Nothing  |    0.2745 ± 0.0625   |    0.3333 ± 0.0660   |    0.2157 ± 0.0576   |



## RAG

To perform the RAG tests, run `testRAG.py` and pass the appropriate `--base_model_path` and `--lora_path` arguments. The `--divide` parameter can be used to decrease the number of testing data point via division.

```bash
# Test RAG performance
python testRAG.py --base_model_path Qwen/Qwen3-4B --lora_path ./models/path/to/lora/checkpoint
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions and support:
- Open an issue on GitHub

---

**FLARKO**: Financial Language-model for Asset Recommendation with Knowledge-graph Optimization

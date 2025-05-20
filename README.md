# Deep Learning-Assignment-3: Hindi-English Transliteration Models

#### Link for the Wandb Report:https://wandb.ai/ma23c014-indian-institute-of-technology-madras/Da6401-Assignment3/reports/Assignment-3--VmlldzoxMjgwNTA1NQ


This repository contains implementations of Sequence-to-Sequence (Seq2Seq) models for Hindi-English transliteration using the Dakshina dataset, with two different approaches:

1. **Vanilla Seq2Seq** - Basic encoder-decoder RNN architecture
2. **Attention-based Seq2Seq** - Enhanced architecture with attention mechanism

## Project Structure

```
├── Attention/
│   ├── predictions_attention/     # Prediction results from attention model
│   ├── Attention.ipynb            # Jupyter notebook with attention implementation
│   ├── allHeatMaps.png            # Visualization of attention weights
│   ├── attention.py               # Python script for attention-based model
├── Vanilla/
│   ├── predictions_vanilla/       # Prediction results from vanilla model
│   ├── Vanilla.ipynb              # Jupyter notebook with vanilla implementation
│   ├── vanilla.py                 # Python script for vanilla Seq2Seq model
├── Question1.py                   # RNN based seq2seq model
└── README.md                      # Main README.md
```

## Overview

This project implements and compares two approaches to transliteration:

1. **Vanilla Seq2Seq**: A standard encoder-decoder architecture using recurrent neural networks (RNN/GRU/LSTM) that encodes the source sequence into a fixed-length vector and decodes it into the target sequence.

2. **Attention-based Seq2Seq**: An enhanced architecture that allows the decoder to focus on different parts of the input sequence during each decoding step, addressing the limitations of compressing all information into a fixed-length vector.

## Task Description

Transliteration is the process of converting text from one script to another while preserving the pronunciation. In this project, we focus on Hindi to Latin script (Romanization) transliteration using the Dakshina dataset.

## Key Features

- **Multiple RNN Cell Types**: Support for RNN, GRU, and LSTM cells
- **Beam Search Decoding**: Enhanced decoding strategy for improved output quality
- **Attention Mechanism**: Visual attention to focus on relevant input characters
- **Hyperparameter Optimization**: Using Weights & Biases (wandb) for tuning
- **Visualization**: Attention weight visualization to understand model behavior

## Requirements

- Python 3.6+
- PyTorch 1.0+
- pandas
- wandb (Weights & Biases)
- tqdm
- matplotlib (for visualizations)

## Installation

```bash
pip install torch pandas wandb tqdm matplotlib
```

## Data

This project uses the Dakshina dataset for Hindi-English transliteration:

```
/kaggle/input/dakshina-dataset/hi/lexicons/
├── hi.translit.sampled.train.tsv
├── hi.translit.sampled.dev.tsv
└── hi.translit.sampled.test.tsv
```

Each file contains tab-separated values with the following format:
```
target_word    source_word    count
```

## Comparison of Models

| Model | Test Accuracy | Inference Speed | Benefits |
|-------|---------------|----------------|----------|
| Vanilla Seq2Seq | ~34.50% | Faster | Simpler architecture, fewer parameters |
| Attention Seq2Seq | ~35.01% | Slower | Better handling of long sequences, improved accuracy |

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Kahkashan2708/Deep_Learning-Assignment-3
   cd Deep_Learning-Assignment-3
   ```

2. Install dependencies:
   ```bash
   pip install torch pandas numpy matplotlib seaborn wandb tqdm
   ```

3. Download the Dakshina dataset from the [official source](https://github.com/google-research-datasets/dakshina) or prepare your own transliteration dataset in the required format.


## Usage

### Vanilla Model

```python
python Vanilla/vanilla.py
```

### Attention Model

```python
python Attention/attention.py
```

### Hyperparameter Optimization

Both models include hyperparameter sweep configurations using Weights & Biases:

```python
sweep_id = wandb.sweep(sweep_config, project="dakshina-transliteration")
wandb.agent(sweep_id, function=main, count=20)
```

## Results Visualization

The attention model generates visualizations of attention weights, showing which input characters the model focuses on when generating each output character:

![Attention Visualization](Attention/allHeatMaps.png)

## Detailed Documentation
# 1. Vanilla Seq2seq:

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Rajnishmaurya/da6401_assignment3
   cd da6401_assignment3
   ```

2. Install dependencies:
   ```bash
   pip install torch pandas numpy matplotlib seaborn wandb tqdm
   ```

3. Download the Dakshina dataset from the [official source](https://github.com/google-research-datasets/dakshina) or prepare your own transliteration dataset in the required format.

## Usage

### Training

To train the model with default parameters:

```python
python vanilla.py
```

### Hyperparameter Optimization

The code includes a hyperparameter sweep configuration using Weights & Biases:

```python
sweep_id = wandb.sweep(sweep_config, project="dakshina-transliteration")
wandb.agent(sweep_id, function=main, count=25)
```

### Evaluation

The model can be evaluated and predictions saved to a CSV file:

```python
evaluate_and_save(model, test_loader, input_vocab, output_vocab, device, csv_path="test_predictions.csv")
```

## Model Architecture

### Encoder

The encoder consists of:
- Embedding layer: Converts input tokens to dense vectors
- RNN layer: Processes the sequence and produces a hidden state

### Decoder

The decoder consists of:
- Embedding layer: Converts output tokens to dense vectors
- RNN layer: Generates output sequence based on encoder hidden state
- Linear layer: Projects RNN outputs to vocabulary space

### Training Process

1. The encoder processes the source sequence and produces a hidden state
2. The decoder uses this hidden state to generate the target sequence
3. Teacher forcing is used to stabilize training
4. Cross-entropy loss is minimized using Adam optimizer

### Inference

During inference, two decoding strategies are available:
- Greedy decoding: Select the most probable token at each step
- Beam search: Maintain multiple hypotheses and select the most probable sequence

## Results

The best model architecture found during experimentation:
- Embedding size: 128
- Hidden size: 128
- Number of layers: 2
- Cell type: LSTM
- Dropout: 0.2
- Learning rate: ~0.00013206
- Batch size: 64

# 2. Attention-Based Neural Transliteration

This repository contains a PyTorch implementation of an attention-based sequence-to-sequence model for transliteration tasks, specifically designed for Hindi-English transliteration using the Dakshina dataset.


## Overview

Transliteration is the process of converting text from one writing system to another while preserving pronunciation. This project implements a neural approach to transliteration using sequence-to-sequence models with Bahdanau attention mechanism.

The implementation provides:
- A flexible encoder-decoder architecture
- Support for multiple RNN cell types (LSTM, GRU, RNN)
- Bahdanau attention mechanism
- Visualization of attention weights
- Hyperparameter tuning with Weights & Biases

## Architecture

The model consists of three main components:

1. **Encoder**: Processes the input sequence and produces a sequence of hidden states.
   - Character-level embedding
   - Multi-layer RNN (configurable: LSTM/GRU/RNN)

2. **Bahdanau Attention**: Computes attention weights for each encoder state at each decoding step.
   - Energy-based alignment model
   - Produces context vectors focusing on relevant input characters

3. **Decoder**: Generates the output sequence one character at a time.
   - Character-level embedding
   - Multi-layer RNN with attention
   - Output projection layer

## Features

- Character-level tokenization and embedding
- Teacher forcing during training
- Dynamic padding and batching
- Attention visualization via heatmaps
- Customizable model hyperparameters
- Support for different RNN cell types
- Hyperparameter optimization with W&B sweeps

## Dataset

This model is trained on the Dakshina dataset, which contains transliteration pairs for various Indian languages. The implementation specifically uses the Hindi-English transliteration pairs.

The dataset should be structured as tab-separated files with the following format:
```
target_language_text \t source_language_text \t count
```

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Rajnishmaurya/da6401_assignment3
   cd da6401_assignment3
   ```

2. Install dependencies:
   ```bash
   pip install torch pandas numpy matplotlib seaborn wandb tqdm
   ```

3. Download the Dakshina dataset from the [official source](https://github.com/google-research-datasets/dakshina) or prepare your own transliteration dataset in the required format.

## Usage

### Training

```python
python train.py --data_path /path/to/dakshina --language hi --epochs 10
```

Optional arguments:
- `--embed_size`: Embedding dimension (default: 128)
- `--hidden_size`: Hidden state dimension (default: 128)
- `--attn_size`: Attention dimension (default: 64)
- `--num_layers`: Number of RNN layers (default: 2)
- `--cell_type`: RNN cell type (LSTM, GRU, RNN) (default: LSTM)
- `--dropout`: Dropout rate (default: 0.3)
- `--batch_size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 0.001)

### Hyperparameter Optimization

```python
python sweep.py --data_path /path/to/dakshina --language hi
```

### Evaluation and Visualization

```python
python evaluate.py --model_path best_model.pth --data_path /path/to/test_data
```


## Attention Visualization

The model includes functionality to visualize attention weights, showing how the decoder attends to different parts of the input sequence at each decoding step:

![Attention Heatmap](allHeatMaps.png)

The attention heatmaps demonstrate how the model focuses on specific input characters when generating each output character, providing insights into the transliteration process.

## Link
[Github Link](https://github.com/Kahkashan2708/Deep_Learning-Assignment-3)  
[Wandb Report](https://wandb.ai/ma23c014-indian-institute-of-technology-madras/Da6401-Assignment3/reports/Assignment-3--VmlldzoxMjgwNTA1NQ)

# Project Structure

This document describes the current structure of the instruct-lite project after reorganization.

## Overview

This repository implements a GPT-2 style language model from scratch using PyTorch, following a simplified educational approach.

## Directory Structure

```
instruct-lite/
├── src/
│   ├── model/              # Model architecture components
│   │   ├── base.py        # Basic building blocks (GELU, FeedForward, LayerNorm)
│   │   ├── GPTConfig.py   # Model configuration
│   │   ├── GPTModel.py    # Main GPT model
│   │   ├── MultiHeadAttention.py  # Multi-head attention implementation
│   │   └── TransformerBlock.py    # Transformer block
│   │
│   ├── loader/             # Weight loading utilities
│   │   └── loadingGPTWeight.py    # Load pretrained GPT-2 weights
│   │
│   ├── finetune/           # Training and evaluation
│   │   ├── training.py    # Training loops
│   │   ├── evaluate.py    # Loss and accuracy calculation
│   │   ├── generate.py    # Text generation functions
│   │   └── graph.py       # Plotting utilities
│   │
│   ├── formatter/          # Prompt formatting utilities (NEW)
│   │   ├── format.py      # Advanced prompt formatters
│   │   └── README.md      # Formatter documentation
│   │
│   ├── rag/                # RAG (Retrieval-Augmented Generation)
│   │   ├── retriever.py   # FAISS-based document retriever
│   │   ├── rag_dataset.py # RAG dataset handling
│   │   ├── rag_agent.py   # RAG agent for inference
│   │   └── train_rag.py   # RAG training loop
│   │
│   └── utils/              # Utilities
│       └── logging.py     # Logging utilities
│
├── notebooks/              # Jupyter notebooks for demos
│   ├── 01_load_weights.ipynb
│   ├── 02_finetune_instruct.ipynb
│   └── 03_rag_demo.ipynb
│
├── data/                   # Dataset directory
│   └── raw/               # Raw datasets
│       ├── alpaca_sample.jsonl
│       └── hotpotqa_sample.jsonl
│
├── scripts/                # Helper scripts
│   ├── export_to_hf.py
│   ├── push_to_hf.sh
│   ├── run_locally.sh
│   └── build_index.sh
│
├── deploy/                 # Deployment files
│   ├── gradio_app.py
│   └── requirements-space.txt
│
├── requirements.txt        # Python dependencies
├── README.md              # Main documentation
└── LICENSE                # MIT License
```

## Core Components

### 1. Model Architecture (`src/model/`)

The model is built from scratch with the following components:

- **base.py**: Basic building blocks
  - `GELU`: GELU activation function
  - `FeedForward`: Position-wise feed-forward network
  - `LayerNorm`: Layer normalization

- **MultiHeadAttention.py**: Multi-head self-attention mechanism with causal masking

- **TransformerBlock.py**: Complete transformer block with attention and feed-forward layers

- **GPTModel.py**: Main GPT model combining embeddings, transformer blocks, and output head

- **GPTConfig.py**: Configuration dictionary for GPT-124M model

### 2. Weight Loading (`src/loader/`)

- **loadingGPTWeight.py**: Functions to load pretrained GPT-2 weights from OpenAI/HuggingFace format

### 3. Training & Evaluation (`src/finetune/`)

- **training.py**: Training loops for both language modeling and classification
  - `train_model_simple()`: Basic training loop
  - `train_classifier_simple()`: Classification training loop
  - `evaluate_model()`: Model evaluation

- **evaluate.py**: Loss and accuracy calculation functions
  - `calc_loss_batch()`: Calculate loss for a single batch
  - `calc_loss_loader()`: Calculate loss over entire data loader
  - Classification variants for sequence classification tasks

- **generate.py**: Text generation utilities
  - `generate()`: Advanced generation with temperature and top-k sampling
  - `generate_text_simple()`: Simple greedy generation
  - `text_to_token_ids()`, `token_ids_to_text()`: Conversion utilities

- **graph.py**: Plotting functions for training curves

### 4. RAG System (`src/rag/`)

**Note**: RAG modules have been updated to work with the new architecture but may need additional testing.

- **retriever.py**: FAISS-based document retrieval
- **rag_agent.py**: Agent for retrieval-augmented generation
- **rag_dataset.py**: Dataset handling for RAG training
- **train_rag.py**: Training loop for RAG models

### 5. Notebooks

- **01_load_weights.ipynb**: Loading and testing pretrained weights
- **02_finetune_instruct.ipynb**: Fine-tuning on instruction data
- **03_rag_demo.ipynb**: RAG demonstration

## Dependencies

The project uses minimal dependencies:
- `torch>=2.2.2`: PyTorch for model implementation
- `tiktoken>=0.5.1`: Tokenization
- `matplotlib>=3.7.1`: Plotting
- `jupyterlab>=4.0`: Notebooks
- `tqdm>=4.66.1`: Progress bars
- `numpy>=1.26`: Numerical operations
- `pandas>=2.2.1`: Data handling
- `tensorflow>=2.16.2`: (Optional, for certain operations)

## Import Structure

All modules can be imported from their respective packages:

```python
# Model components
from src.model import GPTModel, GPT_CONFIG_124M, TransformerBlock

# Weight loading
from src.loader import load_weights_into_gpt

# Training and evaluation
from src.finetune import (
    train_model_simple,
    evaluate_model,
    generate_text_simple,
    text_to_token_ids
)

# RAG (if using)
from src.rag import Retriever, RAGAgent
```

## Recent Changes

### Files Deleted
- Old architecture files (`config.py`, `gpt2_like.py`, `positional_embeddings.py`, `utils.py`)
- Old loader files (`load_gpt2_weights.py`, `checkpoint.py`)
- Old finetune files (`dataset.py`, `train_instruct.py`, `eval_instruct.py`)
- Configuration files (`base_model.yaml`, `instruct_finetune.yaml`, `rag_finetune.yaml`)
- Docker files (`Dockerfile.cpu`, `Dockerfile.gpu`)
- CI/CD files (`github-actions.yml`)
- `pyproject.toml`, `.gitattributes`

### Files Added/Modified
- New modular architecture in `src/model/`
- Simplified loader in `src/loader/`
- Enhanced training utilities in `src/finetune/`
- Updated RAG modules for new architecture
- Simplified `requirements.txt` with core dependencies
- Updated notebooks with correct imports

## Usage Examples

### Load and Initialize Model

```python
import torch
from src.model import GPTModel, GPT_CONFIG_124M

# Initialize model
model = GPTModel(GPT_CONFIG_124M)
model.eval()

print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
```

### Generate Text

```python
import tiktoken
from src.finetune import generate_text_simple, text_to_token_ids, token_ids_to_text

# Initialize tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

# Generate text
input_text = "Once upon a time"
input_ids = text_to_token_ids(input_text, tokenizer)
context_size = GPT_CONFIG_124M["context_length"]

output_ids = generate_text_simple(
    model=model,
    idx=input_ids,
    max_new_tokens=50,
    context_size=context_size
)

print(token_ids_to_text(output_ids, tokenizer))
```

### Train Model

```python
from src.finetune import train_model_simple
from torch.utils.data import DataLoader

# Assuming you have train_loader and val_loader prepared
train_losses, val_losses, tokens_seen = train_model_simple(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    device="cuda",
    num_epochs=3,
    eval_freq=100,
    eval_iter=10,
    start_context="Once upon a time",
    tokenizer=tokenizer
)
```

## Next Steps

1. **Test all modules**: Verify that imports work correctly across all files
2. **Add examples**: Create complete end-to-end examples in notebooks
3. **Document RAG**: Ensure RAG modules work with new tiktoken-based approach
4. **Add tests**: Create unit tests for core functionality
5. **Update deployment**: Adjust deployment scripts if needed

## Notes

- The codebase now uses `tiktoken` instead of HuggingFace transformers for tokenization
- Import paths have been updated from `base.*` to `src.model.*` and `src.finetune.*`
- RAG functionality may need additional adaptation for the new structure
- Some deployment files (Docker, Gradio) may need updates for the new dependencies


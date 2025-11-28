# ChemTransformer

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</p>

A flexible Transformer-based model for **chemical reaction prediction** with compound-aware positional encoding. This implementation introduces an "event-based" SMILES representation that enriches input sequences with auxiliary tokens for reactant roles and reaction conditions.

## ✨ Key Features

- **Compound-Aware Positional Encoding**: Learned positional embeddings that reset at compound boundaries, allowing the model to understand molecular structure within reactions
- **Event-Based Representation**: Support for role tokens (`[REACTANT]`, `[CATALYST]`, `[SOLVENT]`) and condition tokens (temperature, amounts)
- **Standard Transformer Architecture**: 6-layer encoder-decoder with 8 attention heads
- **Modular Design**: Easy to extend for different tokenization strategies (SMILES, SELFIES)

## 📁 Project Structure

```
ChemTransformer/
├── config/
│   └── smile_config.yaml      # Model & training configuration
├── data/
│   ├── reactions_unique.csv.zip   # Sample reaction dataset
│   └── USPTO_480K/                # Place USPTO data here
├── models/
│   ├── __init__.py
│   └── transformer.py         # Transformer with compound positional encoding
├── preprocessing/
│   ├── __init__.py
│   ├── data_split.py          # Train/val/test splitting
│   ├── event2words.py         # Event tokenization
│   └── reaction2events.py     # Reaction to event conversion
├── train/
│   ├── __init__.py
│   └── train.py               # Training script
├── utils/
│   ├── __init__.py
│   └── utils.py               # Utility functions & dataset class
├── scripts/
│   ├── preprocess.sh          # Data preprocessing script
│   ├── train_g2s.sh           # Training script
│   └── validate.sh            # Validation script
└── README.md
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd ChemTransformer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install torch torchvision
pip install rdkit-pypi pandas numpy pyyaml tqdm
```

### 2. Data Preparation

Place your reaction dataset in `data/` directory. The expected format is a CSV file with columns:
- `rxn_smiles`: Full reaction SMILES (reactants>>products)
- Or separate `reactants` and `products` columns

```bash
# Unzip sample data
cd data
unzip reactions_unique.csv.zip
cd ..
```

### 3. Preprocessing

```bash
# Run preprocessing pipeline
sh scripts/preprocess.sh
```

This will:
1. Split data into train/validation/test sets
2. Convert reactions to event sequences
3. Generate vocabulary files

### 4. Training

```bash
# Train the model
sh scripts/train_g2s.sh
```

Or run directly with Python:

```python
python -m train.train --config config/smile_config.yaml
```

### 5. Validation

```bash
# Evaluate on test set
sh scripts/validate.sh
```

## 🏗️ Model Architecture

### Compound Positional Encoding

Unlike standard Transformers that use global positional indices, ChemTransformer uses **compound-aware positional encoding**:

```
Standard:    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, ...]
Ours:        [1, 2, 3, 1, 2, 1, 2, 3, 4, 5, ...]
                 ↑ Mol1  ↑ Mol2  ↑ Mol3
```

This allows the model to understand that positions within each molecule are relative, improving generalization.

### Architecture Details

```python
TransformerModel(
    vocab_size=...,      # Vocabulary size
    d_model=512,         # Model dimension
    nhead=8,             # Number of attention heads
    num_layers=6,        # Encoder/decoder layers
    dim_feedforward=2048,# FFN dimension
    dropout=0.1          # Dropout rate
)
```

## ⚙️ Configuration

Edit `config/smile_config.yaml`:

```yaml
# Model parameters
model:
  d_model: 512
  nhead: 8
  num_layers: 6
  dim_feedforward: 2048
  dropout: 0.1

# Training parameters
training:
  batch_size: 64
  learning_rate: 0.0001
  epochs: 100
  warmup_steps: 4000

# Data parameters
data:
  max_src_len: 512
  max_tgt_len: 512
  vocab_file: data/vocab.txt
```

## 📊 Input Format

### Event-Based Representation

Reactions are converted to event sequences with role annotations:

```
Input:  CC(=O)O.CCO>>CC(=O)OCC
Event:  [REACTANT] C C ( = O ) O [REACTANT] C C O [SEP] [PRODUCT] C C ( = O ) O C C
```

With conditions:

```
[REACTANT] C C ( = O ) O [CATALYST] [Pd/C] [TEMP] 100 [SOLVENT] THF [SEP] [PRODUCT] ...
```

## 📈 Expected Performance

| Dataset | Top-1 Accuracy | Top-5 Accuracy |
|---------|---------------|----------------|
| USPTO-480K | ~88-89% | ~94-95% |

*Performance varies based on data split and hyperparameters.*

## 🔬 Usage Example

```python
import torch
from models.transformer import TransformerModel

# Initialize model
model = TransformerModel(
    vocab_size=5000,
    d_model=512,
    nhead=8,
    num_layers=6
)

# Forward pass
# src: [batch_size, src_len] - tokenized source
# tgt: [batch_size, tgt_len] - tokenized target
# src_pos, tgt_pos: positional indices
output = model(src, tgt, src_pos, tgt_pos, tgt_mask, src_pad_mask, tgt_pad_mask)
```

## 📚 References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Vaswani et al., 2017
- [Molecular Transformer](https://pubs.acs.org/doi/10.1021/acscentsci.9b00576) - Schwaller et al., 2019
- [USPTO Dataset](https://figshare.com/articles/dataset/Chemical_reactions_from_US_patents_1976-Sep2016_/5104873) - Lowe, 2017

## 📝 License

This project is licensed under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

---

*Part of the ChemAI research toolkit for molecular machine learning.*

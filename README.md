# Knowledge Graph Embeddings for Drug-Target Interaction Prediction

A PyTorch implementation of three knowledge graph embedding models (TransE, ComplEx, TriModel) for link prediction and drug-target interaction discovery.

## 📋 Overview

This project implements and compares knowledge graph embedding models on biomedical knowledge graphs, specifically for predicting **drug-target interactions** (DTI). The models learn low-dimensional representations of entities (drugs, proteins, diseases) and relations that can be used to predict missing links in the graph.

### Models Implemented

| Model | Description | Reference |
|-------|-------------|-----------|
| **TransE** | Translating embeddings: $h + r \approx t$ | Bordes et al., 2013 |
| **ComplEx** | Complex-valued embeddings for asymmetric relations | Trouillon et al., 2016 |
| **TriModel** | Tri-vector embeddings with richer interactions | Kamaleldin et al. |

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd Knowledge-Graph-Embeddings-Summary

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
.\venv\Scripts\Activate.ps1
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install torch numpy pandas matplotlib scikit-learn tqdm
```

### Training a Model

```bash
# Train TransE
python TransE_Torch.py

# Train ComplEx
python ComplEx_Torch.py

# Train TriModel
python TriModel_Torch.py
```

### Evaluating a Model

```bash
# Evaluate TransE
python TransE_Torch_evaluation.py

# Evaluate ComplEx
python ComplEx_Torch_evaluation.py

# Evaluate TriModel
python TriModel_Torch_evaluation.py
```

### Compare All Models

```bash
python compare_models.py
```

### Drug-Target Interaction Evaluation

```bash
python evaluate_dti_auc.py
```

### GUI for Predictions

```bash
python predict_fact_gui.py
```

## 📁 Project Structure

```
Knowledge-Graph-Embeddings-Summary/
│
├── model.py                      # Core model definitions (TransE, ComplEx, TriModel)
│
├── TransE_Torch.py               # TransE training script
├── ComplEx_Torch.py              # ComplEx training script
├── TriModel_Torch.py             # TriModel training script
│
├── TransE_Torch_evaluation.py    # TransE evaluation (MRR, Hits@k)
├── ComplEx_Torch_evaluation.py   # ComplEx evaluation
├── TriModel_Torch_evaluation.py  # TriModel evaluation
│
├── compare_models.py             # Compare all models, generate visualizations
├── evaluate_dti_auc.py           # DTI-specific evaluation (AUC-ROC, AUC-PR)
│
├── predict_fact.py               # Command-line prediction tool
├── predict_fact_gui.py           # GUI for interactive predictions
│
├── data/
│   ├── transe/                   # Train/valid/test splits for TransE
│   ├── complex/                  # Train/valid/test splits for ComplEx
│   └── trimodel/                 # Train/valid/test splits for TriModel
│
├── outputs_transe/               # TransE trained model and results
├── outputs_complex/              # ComplEx trained model and results
├── outputs_trimodel/             # TriModel trained model and results
├── outputs_comparison/           # Comparison charts and reports
├── outputs_dti_evaluation/       # DTI evaluation results
│
├── drugbank_facts.txt            # DrugBank knowledge graph data
├── packages_summary.txt          # Package dependencies documentation
└── README.md                     # This file
```

## 📊 Results

### Link Prediction (Filtered Ranking)

| Model | MRR | Hits@1 | Hits@3 | Hits@10 |
|-------|-----|--------|--------|---------|
| TransE | 0.3828 | 0.3216 | 0.4120 | 0.4899 |
| ComplEx | 0.5521 | 0.4923 | 0.5797 | 0.6645 |
| **TriModel** | **0.6158** | **0.5539** | **0.6427** | **0.7378** |

### Drug-Target Interaction Prediction

| Model | AUC-ROC | AUC-PR | Best F1 | Precision | Recall |
|-------|---------|--------|---------|-----------|--------|
| TransE | 0.7922 | 0.2445 | 0.2954 | 0.3776 | 0.2426 |
| **ComplEx** | 0.8543 | **0.4682** | **0.4973** | **0.6054** | **0.4219** |
| TriModel | **0.8548** | 0.4431 | 0.4774 | 0.5813 | 0.4050 |

**Key Findings:**
- **TriModel** achieves the best overall link prediction performance (highest MRR and Hits@k)
- **ComplEx** is best for DTI prediction (highest AUC-PR and F1 score)
- **TransE** is simpler but underperforms on this biomedical dataset

## 🔧 Configuration

Training hyperparameters can be modified in each training script:

```python
# Example configuration
EMBEDDING_DIM = 200      # Embedding dimension
LEARNING_RATE = 0.001    # Learning rate
BATCH_SIZE = 1024        # Training batch size
NUM_EPOCHS = 100         # Number of training epochs
MARGIN = 1.0             # Margin for ranking loss
NEG_SAMPLES = 10         # Negative samples per positive
```

## 📈 Evaluation Metrics

### Link Prediction
- **MRR** (Mean Reciprocal Rank): Average of 1/rank for correct predictions
- **Hits@k**: Proportion of correct entities ranked in top k

### DTI Prediction
- **AUC-ROC**: Area under ROC curve
- **AUC-PR**: Area under Precision-Recall curve (better for imbalanced data)
- **F1 Score**: Harmonic mean of precision and recall

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| torch | >= 1.9 | Deep learning framework |
| numpy | >= 1.20 | Numerical computations |
| pandas | >= 1.3 | Data manipulation |
| matplotlib | >= 3.4 | Visualization |
| scikit-learn | >= 0.24 | Evaluation metrics |
| tqdm | >= 4.60 | Progress bars |

## 📚 References

1. Bordes, A., et al. "Translating Embeddings for Modeling Multi-relational Data." NIPS 2013.
2. Trouillon, T., et al. "Complex Embeddings for Simple Link Prediction." ICML 2016.
3. [libkge](https://github.com/samehkamaleldin/libkge) - Knowledge Graph Embedding Library

## 📄 License

This project is for educational and research purposes.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

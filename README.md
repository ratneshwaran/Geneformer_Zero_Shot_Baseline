# Dissertation Research: Single-Cell Foundation Model Baselines

## Overview

This repository is a fork of Microsoft's [zero-shot-scfoundation](https://github.com/microsoft/zero-shot-scfoundation) repository, adapted for dissertation research on single-cell RNA sequencing (scRNA-seq) foundation models. The original repository accompanies the paper "Assessing the limits of zero-shot foundation models in single-cell biology" and provides evaluation frameworks for Geneformer and scGPT models.

## Research Purpose

This adapted version is being used to establish baseline performance metrics for foundation models (Geneformer V2 CLcancer and scGPT) across multiple cancer datasets as part of dissertation research. The focus is on evaluating zero-shot capabilities for:

1. **Binary malignancy classification** - Distinguishing malignant vs. non-malignant cells
2. **Cell type annotation** - Multi-class cell type identification
3. **Cross-dataset generalization** - Performance across different cancer types

## Datasets Used

The following cancer datasets are being evaluated:

### Binary Malignancy Classification
- **Ovarian Cancer** (`Geneformer_binary_ovarian_malignancy.ipynb`)
- **Pancreatic Cancer** (`Geneformer_binary_pancreas_malignancy.ipynb`) 
- **Prostate Cancer** (`Geneformer_binary_prostate_malignancy.ipynb`)
- **Kidney Cancer** (`Geneformer_binary_kidney_malignancy.ipynb`)
- **Integrated Multi-Cancer (Breast)** (`Geneformer_binary_integrated_malignancy.ipynb`)

### Multi-Class Cell Type Classification
- **Ovarian Cancer** (`Geneformer_zero_shot_ovarian_baseline.ipynb`)
- **Pancreatic Cancer** (`Geneformer_zero_shot_pancreas_baseline.ipynb`)
- **Prostate Cancer** (`Geneformer_zero_shot_prostate_baseline.ipynb`)
- **Kidney Cancer** (`Geneformer_zero_shot_kidney_baseline.ipynb`)
- **Integrated Multi-Cancer (Breast)** (`Geneformer_zero_shot_integrated_baseline.ipynb`)

## Key Modifications

### Enhanced Evaluation Framework
- **Binary classification pipelines** for malignancy detection
- **Ensemble evaluation metrics** combining embeddings and gene expression predictions
- **Cross-dataset evaluation** capabilities for generalization testing
- **Robust error handling** for diverse dataset characteristics

### Custom Evaluation Metrics
- **Classification metrics**: Accuracy, Precision, Recall, F1-Score, AUC-ROC
- **Embedding quality**: Silhouette scores, graph connectivity
- **Batch integration**: Principal component regression (PCR)
- **Clustering metrics**: Normalized Mutual Information (NMI), Adjusted Rand Index (ARI)

### Data Processing Enhancements
- **Automated binary label generation** from cell type annotations using regex patterns
- **Data validation and cleaning** to handle missing values and edge cases
- **Consistent preprocessing** across multiple cancer datasets
- **Tokenization and embedding extraction** optimized for Geneformer V2 CLcancer model
- **Cancer-specific model evaluation** using the specialized Geneformer variant trained on cancer data

## Repository Structure

```
notebooks/
├── Geneformer_binary_*_malignancy.ipynb    # Binary classification notebooks
├── Geneformer_zero_shot_*_baseline.ipynb   # Multi-class classification notebooks
└── load_geneformer_results.py              # Results analysis utilities

sc_foundation_evals/
├── cell_embeddings.py                      # Cell embedding evaluation
├── geneformer_ensemble_metrics.py          # Ensemble evaluation framework
├── geneformer_forward.py                   # Geneformer model interface
└── utils.py                               # Utility functions with enhancements

output/
└── geneformer/                            # Model outputs and results
    ├── ovarian_binary/
    ├── pancreas_binary/
    ├── prostate_binary/
    ├── kidney_binary/
    └── integrated_binary/
```

## Usage

### Running Binary Malignancy Classification

```bash
# Activate the environment
conda activate sc_foundation_evals

# Run binary classification for specific cancer type
jupyter notebook notebooks/Geneformer_binary_ovarian_malignancy.ipynb
```

### Running Multi-Class Cell Type Classification

```bash
# Run multi-class classification for specific cancer type
jupyter notebook notebooks/Geneformer_zero_shot_ovarian_baseline.ipynb
```

### Analyzing Results

```bash
# Load and analyze results across datasets
python notebooks/load_geneformer_results.py
```

## Key Research Findings

- **Dataset-specific performance variations** in malignancy detection
- **Embedding quality metrics** across different cancer types
- **Generalization capabilities** when training on one cancer type and testing on others
- **Comparison with baseline methods** (scVI, HVG selection)

## Technical Improvements Made

1. **Error Handling**: Enhanced robustness for datasets with missing labels or single-class distributions
2. **Memory Optimization**: Efficient processing of large single-cell datasets
3. **Evaluation Metrics**: Comprehensive evaluation framework for both binary and multi-class tasks
4. **Data Validation**: Automated checks for data quality and preprocessing consistency

## Original Repository Credit

This work builds upon the excellent foundation provided by:

**Original Repository**: [microsoft/zero-shot-scfoundation](https://github.com/microsoft/zero-shot-scfoundation)

**Original Paper**: "Assessing the limits of zero-shot foundation models in single-cell biology"  
DOI: [10.1101/2023.10.16.561085](https://www.biorxiv.org/content/10.1101/2023.10.16.561085)

**Original Authors**: Kasia Kedzierska, et al.

## Dependencies

The repository maintains compatibility with the original dependencies:

- Python 3.9+
- PyTorch 1.13+
- CUDA 11.7+
- FlashAttention v1.0.4
- scGPT v0.1.6
- Geneformer V2 CLcancer model (Geneformer-V2-104M_CLcancer)
- scIB v1.0.4

## Installation

Follow the original installation instructions in the main README.md, then clone this adapted version:

```bash
git clone [your-fork-url]
cd zero-shot-scfoundation
```

## Dissertation Context

This work is part of dissertation research investigating the application of foundation models to cancer biology, specifically focusing on:

- **Cross-cancer generalizability** of single-cell foundation models
- **Binary vs. multi-class classification** performance comparison
- **Embedding quality assessment** across diverse cancer datasets
- **Baseline establishment** for future model development

## License

This adapted version maintains the original MIT license from the Microsoft repository while acknowledging the derivative nature of the work.

# Geneformer Zero-Shot Pancreas Cancer Baseline

This directory contains the Geneformer zero-shot pancreas cancer baseline implementation for comparison with LangCell results.

## Overview

The Geneformer baseline notebook (`Geneformer_zero_shot_pancreas_baseline.ipynb`) creates a comprehensive baseline for comparing Geneformer performance with LangCell on pancreas cancer data. The notebook:

1. **Loads and preprocesses** pancreas cancer data
2. **Extracts embeddings** using Geneformer-V2-104M_CLcancer model
3. **Performs comprehensive evaluation** with multiple metrics
4. **Saves results** for future inferences (similar to LangCell notebooks)
5. **Creates visualizations** and comparison plots

## Files

- `Geneformer_zero_shot_pancreas_baseline.ipynb` - Main baseline notebook
- `load_geneformer_results.py` - Utility script for loading and comparing results
- `README_Geneformer_Baseline.md` - This file

## Usage

### 1. Run the Baseline Notebook

```bash
# Navigate to the notebooks directory
cd zero-shot-scfoundation/notebooks

# Run the baseline notebook
jupyter notebook Geneformer_zero_shot_pancreas_baseline.ipynb
```

The notebook will:
- Set up cache directories
- Load the Geneformer model
- Process pancreas cancer data
- Extract embeddings
- Perform evaluation
- Save results to `../output/geneformer/pancreas_baseline/model_outputs/`

### 2. Saved Results

The notebook saves the following files:

- `geneformer_pancreas_results.pt` - Main results file (similar to LangCell format)
- `geneformer_pancreas_metadata.json` - Metadata about the evaluation
- `geneformer_pancreas_metrics.csv` - Detailed metrics for different alpha values
- `geneformer_pancreas_confusion_matrix.png` - Confusion matrix visualization
- `geneformer_pancreas_performance.png` - Performance comparison plot

### 3. Load Results for Future Use

```python
# Load saved results
results = torch.load('../output/geneformer/pancreas_baseline/model_outputs/geneformer_pancreas_results.pt')

# Access results
cell_embs = results['cell_embs']  # Cell embeddings
predictions = results['preds']     # Predicted labels
labels = results['labels']         # True labels
class_names = results['class_names']  # Class names
best_alpha = results['best_alpha']    # Best alpha value
```

### 4. Compare with LangCell Results

Use the utility script to compare results:

```bash
# Run the comparison script
python load_geneformer_results.py
```

Or use it in your own code:

```python
from load_geneformer_results import load_geneformer_results, load_langcell_results, compare_results

# Load results
geneformer_results = load_geneformer_results('path/to/geneformer_results.pt')
langcell_results = load_langcell_results('path/to/langcell_results.pt')

# Compare results
comparison = compare_results(geneformer_results, langcell_results, output_dir='../output/comparisons/')
```

## Results Format

The saved results follow a similar format to LangCell notebooks:

```python
results = {
    'cell_embs': torch.Tensor,      # Cell embeddings (n_cells, embedding_dim)
    'logits': torch.Tensor,         # Prediction scores (n_cells, n_classes)
    'preds': torch.Tensor,          # Predicted labels (n_cells,)
    'labels': torch.Tensor,         # True labels (n_cells,)
    'class_names': list,            # Class names
    'best_alpha': float,            # Best alpha value for ensemble
    'model_name': str,              # Model name
    'dataset_name': str,            # Dataset name
    'embedding_dim': int,           # Embedding dimension
    'n_cells': int,                 # Number of cells
    'n_classes': int,               # Number of classes
    'evaluation_date': str,         # Evaluation date
    'evaluation_metrics': dict      # Evaluation metrics
}
```

## Key Differences from Original Notebook

1. **Comprehensive result saving** - Results are saved in a format compatible with LangCell notebooks
2. **Enhanced evaluation** - Multiple evaluation metrics and visualizations
3. **Baseline comparison** - Designed specifically for comparison with LangCell
4. **Better organization** - Clear sections and documentation
5. **Utility functions** - Helper functions for loading and comparing results

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce `batch_size` in the notebook
2. **Missing data**: Ensure the pancreas dataset is available at the specified path
3. **Model not found**: Check that the Geneformer model is properly installed

### Data Requirements

- Pancreas cancer dataset: `../data/datasets/integrated_with_quiescence.h5ad`
- Geneformer model: `../Geneformer/Geneformer-V2-104M_CLcancer`
- Geneformer dictionaries: `../Geneformer/geneformer`

## Performance Metrics

The baseline typically achieves:
- **F1-Score**: ~0.53
- **AUC-ROC**: ~0.77
- **Accuracy**: ~0.49

These metrics provide a baseline for comparison with LangCell performance on the same dataset.

## Comparison with LangCell

To compare with LangCell results:

1. Run the LangCell pancreas notebook
2. Run this Geneformer baseline notebook
3. Use the comparison script to analyze differences
4. Review the generated comparison plots and metrics

The comparison will show:
- Overall performance differences
- Per-class performance analysis
- Visualization of results
- Detailed metrics comparison

## Citation

If you use this baseline in your research, please cite:

```bibtex
@article{geneformer2023,
  title={Geneformer: Learned gene compression using transformer-based context modeling},
  author={Theodoris, Christina V and Xiao, Ling and Chopra, Anant and Chaffin, Mark D and Al Sayed, Zeina R and Hill, Matthew C and Mantineo, Helene and Brydon, Elizabeth M and Zeng, Zexian and Liu, X Shirley and others},
  journal={Nature},
  volume={618},
  number={7966},
  pages={1--9},
  year={2023},
  publisher={Nature Publishing Group UK London}
}
``` 
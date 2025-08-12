# Geneformer Zero-Shot Pancreas Notebook - Fixes and Improvements

## Current Issues and Solutions

### 1. **Missing Result Saving Functionality**

**Issue**: The current notebook doesn't save results for future inferences like LangCell notebooks.

**Solution**: Add comprehensive result saving functionality:

```python
# Add this after the evaluation section
import torch
import json
from datetime import datetime

# Save results for future inferences
results_dict = {
    'cell_embs': torch.tensor(em.embeddings, dtype=torch.float32),
    'logits': torch.tensor(prediction_scores, dtype=torch.float32),
    'preds': torch.tensor(predictions, dtype=torch.long),
    'labels': torch.tensor(em.labels_encoded, dtype=torch.long),
    'class_names': em.class_names,
    'best_alpha': best_alpha,
    'model_name': 'Geneformer-V2-104M_CLcancer',
    'dataset_name': dataset_name,
    'embedding_dim': em.embeddings.shape[1],
    'n_cells': len(em.embeddings),
    'n_classes': len(em.class_names),
    'evaluation_date': datetime.now().isoformat(),
    'evaluation_metrics': best_result
}

# Save to disk
results_path = os.path.join(model_out, 'geneformer_pancreas_results.pt')
torch.save(results_dict, results_path)
```

### 2. **Incomplete Evaluation Metrics**

**Issue**: The notebook only shows basic metrics without comprehensive analysis.

**Solution**: Add comprehensive evaluation:

```python
# Add comprehensive evaluation metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    average_precision_score
)

# Evaluate with different alpha values
alphas = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
results = {}

for alpha in alphas:
    try:
        result = em.evaluate_ensemble(alpha=alpha, method='embedding_only')
        results[alpha] = result
        print(f"Alpha {alpha}: F1={result['f1_score']:.3f}, AUC-ROC={result['auc_roc']:.3f}")
    except Exception as e:
        print(f"Alpha {alpha}: Failed - {e}")
```

### 3. **Missing Visualizations**

**Issue**: No visualizations for results analysis.

**Solution**: Add visualization code:

```python
# Create confusion matrix
import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(em.labels_encoded, predictions)
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
           xticklabels=em.class_names, 
           yticklabels=em.class_names, ax=ax)
ax.set_title(f'Geneformer Pancreas Cancer - Confusion Matrix (α={best_alpha})')
plt.tight_layout()
plt.savefig(os.path.join(model_out, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
plt.close()
```

### 4. **No Baseline Comparison Structure**

**Issue**: The notebook isn't designed for comparison with LangCell.

**Solution**: Restructure for baseline comparison:

```python
# Add baseline comparison structure
print("="*80)
print("GENEFORMER PANCREAS CANCER BASELINE EVALUATION")
print("="*80)
print(f"Dataset: {dataset_name}")
print(f"Model: Geneformer-V2-104M_CLcancer")
print(f"Total cells: {len(em.embeddings)}")
print(f"Cell types: {len(em.class_names)}")
print(f"Best F1-Score: {best_result['f1_score']:.3f}")
print(f"Best AUC-ROC: {best_result['auc_roc']:.3f}")
print("="*80)
```

## Complete Fixed Notebook Structure

### 1. **Setup and Imports**
```python
# Enhanced imports
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    average_precision_score
)
```

### 2. **Configuration**
```python
# Enhanced configuration
output_dir = "../output/geneformer/pancreas_baseline/"
model_out = os.path.join(output_dir, "model_outputs")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(model_out, exist_ok=True)
```

### 3. **Data Loading and Preprocessing**
```python
# Enhanced data loading with error handling
try:
    input_data = data.InputData(adata_dataset_path = in_dataset_path)
    input_data.preprocess_data(gene_col = gene_col,
                               model_type = "geneformer",
                               save_ext = "loom",
                               gene_name_id_dict = geneform.gene_name_id,
                               preprocessed_path = preprocessed_path)
except Exception as e:
    print(f"Error in data preprocessing: {e}")
    raise
```

### 4. **Model Evaluation**
```python
# Enhanced evaluation with comprehensive metrics
print("Performing comprehensive evaluation...")
eval_ce.evaluate(n_cells = 1000, embedding_key = "geneformer")

# Create ensemble evaluation
em = geneformer_ensemble_metrics.create_geneformer_ensemble_evaluation(
    geneform_instance=geneform,
    input_data=input_data,
    label_col=label_col,
    output_dir=model_out
)
```

### 5. **Result Saving**
```python
# Comprehensive result saving
results_dict = {
    'cell_embs': torch.tensor(em.embeddings, dtype=torch.float32),
    'logits': torch.tensor(prediction_scores, dtype=torch.float32),
    'preds': torch.tensor(predictions, dtype=torch.long),
    'labels': torch.tensor(em.labels_encoded, dtype=torch.long),
    'class_names': em.class_names,
    'best_alpha': best_alpha,
    'model_name': 'Geneformer-V2-104M_CLcancer',
    'dataset_name': dataset_name,
    'embedding_dim': em.embeddings.shape[1],
    'n_cells': len(em.embeddings),
    'n_classes': len(em.class_names),
    'evaluation_date': datetime.now().isoformat(),
    'evaluation_metrics': best_result
}

# Save results
results_path = os.path.join(model_out, 'geneformer_pancreas_results.pt')
torch.save(results_dict, results_path)

# Save metadata
metadata = {
    'model_name': 'Geneformer-V2-104M_CLcancer',
    'dataset_name': dataset_name,
    'best_alpha': best_alpha,
    'evaluation_metrics': best_result,
    'class_names': em.class_names,
    'embedding_dim': em.embeddings.shape[1],
    'n_cells': len(em.embeddings),
    'n_classes': len(em.class_names),
    'evaluation_date': datetime.now().isoformat()
}

metadata_path = os.path.join(model_out, 'geneformer_pancreas_metadata.json')
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)
```

### 6. **Visualizations**
```python
# Create visualizations
# Confusion matrix
cm = confusion_matrix(em.labels_encoded, predictions)
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
           xticklabels=em.class_names, 
           yticklabels=em.class_names, ax=ax)
ax.set_title(f'Geneformer Pancreas Cancer - Confusion Matrix (α={best_alpha})')
plt.tight_layout()
plt.savefig(os.path.join(model_out, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
plt.close()

# Performance comparison plot
fig, ax = plt.subplots(figsize=(10, 6))
alphas_list = list(results.keys())
f1_scores = [results[alpha]['f1_score'] for alpha in alphas_list]
auc_scores = [results[alpha]['auc_roc'] for alpha in alphas_list]

ax.plot(alphas_list, f1_scores, 'o-', label='F1-Score', linewidth=2, markersize=8)
ax.plot(alphas_list, auc_scores, 's-', label='AUC-ROC', linewidth=2, markersize=8)
ax.set_xlabel('Alpha (Weight for Embeddings)')
ax.set_ylabel('Score')
ax.set_title('Geneformer Pancreas Cancer - Performance Comparison')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(model_out, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()
```

## Key Improvements Made

1. **Comprehensive Result Saving**: Results are saved in a format compatible with LangCell notebooks
2. **Enhanced Evaluation**: Multiple evaluation metrics and comprehensive analysis
3. **Visualizations**: Confusion matrix and performance comparison plots
4. **Error Handling**: Better error handling and logging
5. **Baseline Structure**: Designed specifically for comparison with LangCell
6. **Documentation**: Clear documentation and comments
7. **Metadata**: Additional metadata for reproducibility

## Usage Instructions

1. **Run the fixed notebook**: `Geneformer_zero_shot_pancreas_baseline.ipynb`
2. **Check results**: Look in `../output/geneformer/pancreas_baseline/model_outputs/`
3. **Compare with LangCell**: Use the comparison script `load_geneformer_results.py`
4. **Analyze results**: Review the generated plots and metrics

## Expected Performance

Based on the current implementation, the Geneformer baseline typically achieves:
- **F1-Score**: ~0.53
- **AUC-ROC**: ~0.77
- **Accuracy**: ~0.49

These metrics provide a solid baseline for comparison with LangCell performance on the same pancreas cancer dataset. 
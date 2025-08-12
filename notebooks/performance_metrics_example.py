# Performance Metrics Example for Geneformer Evaluation
# Add this to your notebook after extracting embeddings

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import the performance metrics module
from sc_foundation_evals import performance_metrics

# After you have extracted embeddings with geneform.extract_embeddings()
# Create performance evaluation object
pm = performance_metrics.create_performance_evaluation(
    geneform_instance=geneform,
    input_data=input_data,
    label_col=label_col,  # Your cell type column name
    output_dir=model_out  # Your output directory
)

# Evaluate with all classifiers
print("Evaluating embeddings with multiple classifiers...")
results = pm.evaluate_all_classifiers(test_size=0.2)

# Print summary
pm.print_summary()

# Create visualizations
print("\nCreating visualizations...")

# 1. Confusion matrix for best classifier
best_classifier, _ = pm.get_best_classifier()
fig1 = pm.plot_confusion_matrix(classifier_type=best_classifier)
plt.show()

# 2. Metrics comparison across classifiers
fig2 = pm.plot_metrics_comparison()
plt.show()

# 3. Per-class metrics for best classifier
fig3 = pm.plot_per_class_metrics(classifier_type=best_classifier)
plt.show()

# Save results
pm.save_results()

# Get detailed results for specific classifier
if 'random_forest' in results:
    rf_results = results['random_forest']
    print(f"\nRandom Forest Results:")
    print(f"Accuracy: {rf_results['accuracy']:.3f}")
    print(f"Precision: {rf_results['precision']:.3f}")
    print(f"Recall: {rf_results['recall']:.3f}")
    print(f"F1-Score: {rf_results['f1_score']:.3f}")
    
    # Per-class F1 scores
    print(f"\nPer-class F1 scores:")
    for class_name, f1_score in rf_results['per_class_f1'].items():
        print(f"  {class_name}: {f1_score:.3f}")

# Additional analysis: Embedding quality assessment
print(f"\nEmbedding Quality Assessment:")
print(f"Number of cells: {len(pm.embeddings)}")
print(f"Embedding dimension: {pm.embeddings.shape[1]}")
print(f"Number of cell types: {len(pm.class_names)}")
print(f"Cell types: {list(pm.class_names)}")

# Calculate embedding statistics
embedding_mean = np.mean(pm.embeddings, axis=0)
embedding_std = np.std(pm.embeddings, axis=0)
print(f"Embedding mean: {np.mean(embedding_mean):.4f}")
print(f"Embedding std: {np.mean(embedding_std):.4f}")

# Class distribution
unique, counts = np.unique(pm.labels, return_counts=True)
print(f"\nClass distribution:")
for class_name, count in zip(unique, counts):
    print(f"  {class_name}: {count} cells ({count/len(pm.labels)*100:.1f}%)") 
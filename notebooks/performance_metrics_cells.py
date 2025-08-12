# Performance Metrics Cells for Geneformer Notebook (Updated with AUC-ROC and Average Precision)
# Copy these cells into your notebook after the existing evaluation cells

# Cell 41: Import performance metrics
"""
# Import performance metrics module
from sc_foundation_evals import performance_metrics

print("Performance metrics module imported successfully!")
"""

# Cell 42: Create performance evaluation object
"""
# Create performance evaluation object
pm = performance_metrics.create_performance_evaluation(
    geneform_instance=geneform,
    input_data=input_data,
    label_col=label_col,  # Your cell type column name
    output_dir=model_out  # Your output directory
)

print("Performance evaluation object created!")
print(f"Number of cells: {len(pm.embeddings)}")
print(f"Embedding dimension: {pm.embeddings.shape[1]}")
print(f"Number of cell types: {len(pm.class_names)}")
print(f"Cell types: {list(pm.class_names)}")
"""

# Cell 43: Evaluate with all classifiers
"""
# Evaluate with all classifiers
print("Evaluating embeddings with multiple classifiers...")
results = pm.evaluate_all_classifiers(test_size=0.2)

print("Evaluation complete!")
"""

# Cell 44: Print summary
"""
# Print comprehensive summary
pm.print_summary()
"""

# Cell 45: Confusion matrix
"""
# Create visualizations
print("Creating performance visualizations...")

# 1. Confusion matrix for best classifier
best_classifier, _ = pm.get_best_classifier()
print(f"Best classifier: {best_classifier}")

fig1 = pm.plot_confusion_matrix(classifier_type=best_classifier)
plt.show()
"""

# Cell 46: Metrics comparison
"""
# 2. Metrics comparison across classifiers
fig2 = pm.plot_metrics_comparison()
plt.show()
"""

# Cell 47: Per-class metrics
"""
# 3. Per-class metrics for best classifier
fig3 = pm.plot_per_class_metrics(classifier_type=best_classifier)
plt.show()
"""

# Cell 48: ROC curves
"""
# 4. ROC curves for best classifier
fig4 = pm.plot_roc_curves(classifier_type=best_classifier)
plt.show()
"""

# Cell 49: Precision-Recall curves
"""
# 5. Precision-Recall curves for best classifier
fig5 = pm.plot_precision_recall_curves(classifier_type=best_classifier)
plt.show()
"""

# Cell 50: Save results
"""
# Save results to CSV
pm.save_results(filename='geneformer_performance_metrics.csv')
print("Results saved to CSV file!")
"""

# Cell 51: Detailed Random Forest analysis
"""
# Detailed analysis for Random Forest (if available)
if 'random_forest' in results:
    rf_results = results['random_forest']
    print(f"\nRandom Forest Detailed Results:")
    print(f"Accuracy:      {rf_results['accuracy']:.3f}")
    print(f"Precision:     {rf_results['precision']:.3f}")
    print(f"Recall:        {rf_results['recall']:.3f}")
    print(f"F1-Score:      {rf_results['f1_score']:.3f}")
    
    # Add AUC-ROC and Average Precision if available
    if 'auc_roc' in rf_results and rf_results['auc_roc'] is not None:
        print(f"AUC-ROC:       {rf_results['auc_roc']:.3f}")
    if 'avg_precision' in rf_results and rf_results['avg_precision'] is not None:
        print(f"Avg Precision: {rf_results['avg_precision']:.3f}")
    
    # Per-class F1 scores
    print(f"\nPer-class F1 scores:")
    for class_name, f1_score in rf_results['per_class_f1'].items():
        print(f"  {class_name}: {f1_score:.3f}")
    
    # Per-class AUC-ROC scores (if available)
    if 'per_class_auc_roc' in rf_results:
        print(f"\nPer-class AUC-ROC scores:")
        for class_name, auc_roc in rf_results['per_class_auc_roc'].items():
            if auc_roc is not None:
                print(f"  {class_name}: {auc_roc:.3f}")
    
    # Per-class Average Precision scores (if available)
    if 'per_class_avg_precision' in rf_results:
        print(f"\nPer-class Average Precision scores:")
        for class_name, avg_precision in rf_results['per_class_avg_precision'].items():
            if avg_precision is not None:
                print(f"  {class_name}: {avg_precision:.3f}")
else:
    print("Random Forest results not available")
"""

# Cell 52: Embedding quality assessment
"""
# Additional embedding quality assessment
print(f"\nEmbedding Quality Assessment:")
print(f"Number of cells: {len(pm.embeddings)}")
print(f"Embedding dimension: {pm.embeddings.shape[1]}")
print(f"Number of cell types: {len(pm.class_names)}")

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
"""

# Cell 53: Test size comparison
"""
# Performance comparison with different test sizes
print("Testing performance with different test sizes...")

test_sizes = [0.1, 0.2, 0.3]
test_results = {}

for test_size in test_sizes:
    print(f"\nTesting with {test_size*100}% test size...")
    try:
        results = pm.evaluate_with_classifier('random_forest', test_size=test_size)
        test_results[test_size] = {
            'f1_score': results['f1_score'],
            'auc_roc': results.get('auc_roc'),
            'avg_precision': results.get('avg_precision')
        }
        print(f"F1-Score: {results['f1_score']:.3f}")
        if results.get('auc_roc') is not None:
            print(f"AUC-ROC: {results['auc_roc']:.3f}")
        if results.get('avg_precision') is not None:
            print(f"Avg Precision: {results['avg_precision']:.3f}")
    except Exception as e:
        print(f"Error with test_size {test_size}: {e}")

if test_results:
    print(f"\nTest size comparison:")
    for test_size, metrics in test_results.items():
        print(f"  {test_size*100}% test: F1={metrics['f1_score']:.3f}", end="")
        if metrics['auc_roc'] is not None:
            print(f", AUC-ROC={metrics['auc_roc']:.3f}", end="")
        if metrics['avg_precision'] is not None:
            print(f", AP={metrics['avg_precision']:.3f}", end="")
        print()
"""

# Cell 54: Final summary
"""
# Summary of all evaluations
print("\n" + "="*80)
print("GENEFORMER PERFORMANCE EVALUATION SUMMARY")
print("="*80)

print(f"Dataset: {dataset_name}")
print(f"Model: Geneformer-V2-104M_CLcancer")
print(f"Total cells: {len(pm.embeddings)}")
print(f"Cell types: {len(pm.class_names)}")
print(f"Embedding dimension: {pm.embeddings.shape[1]}")

if results:
    best_classifier, best_results = pm.get_best_classifier()
    print(f"\nBest performing classifier: {best_classifier}")
    print(f"Best F1-Score: {best_results['f1_score']:.3f}")
    print(f"Best Accuracy: {best_results['accuracy']:.3f}")
    print(f"Best Precision: {best_results['precision']:.3f}")
    print(f"Best Recall: {best_results['recall']:.3f}")
    
    # Add AUC-ROC and Average Precision if available
    if 'auc_roc' in best_results and best_results['auc_roc'] is not None:
        print(f"Best AUC-ROC: {best_results['auc_roc']:.3f}")
    if 'avg_precision' in best_results and best_results['avg_precision'] is not None:
        print(f"Best Avg Precision: {best_results['avg_precision']:.3f}")

print("\nResults saved to:")
print(f"  - CSV: {model_out}/geneformer_performance_metrics.csv")
print(f"  - Plots: {model_out}/confusion_matrix_*.png")
print(f"  - Plots: {model_out}/metrics_comparison.png")
print(f"  - Plots: {model_out}/per_class_metrics_*.png")
print(f"  - Plots: {model_out}/roc_curves_*.png")
print(f"  - Plots: {model_out}/precision_recall_curves_*.png")

print("="*80)
""" 
# Ensemble Metrics Cells for Geneformer Notebook (LangCell-style approach)
# Copy these cells into your notebook after extracting embeddings

# Cell 55: Import ensemble metrics
"""
# Import ensemble metrics module (LangCell-style approach)
from sc_foundation_evals import geneformer_ensemble_metrics

print("Ensemble metrics module imported successfully!")
"""

# Cell 56: Create ensemble evaluation object
"""
# Create ensemble evaluation object (similar to LangCell approach)
em = geneformer_ensemble_metrics.create_geneformer_ensemble_evaluation(
    geneform_instance=geneform,
    input_data=input_data,
    label_col=label_col,  # Your cell type column name
    output_dir=model_out  # Your output directory
)

print("Ensemble evaluation object created!")
print(f"Number of cells: {len(em.embeddings)}")
print(f"Embedding dimension: {em.embeddings.shape[1]}")
print(f"Number of cell types: {len(em.class_names)}")
print(f"Input rankings length: {len(em.input_rankings)}")
print(f"Output rankings length: {len(em.output_rankings)}")
"""

# Cell 57: Evaluate with different alpha values
"""
# Evaluate ensemble with different alpha values (similar to LangCell)
print("Evaluating ensemble with different alpha values...")

alphas = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
results = em.evaluate_multiple_alphas(alphas, method='embedding_ranking')

print("Evaluation complete!")
print(f"Tested {len(results)} alpha values")
"""

# Cell 58: Print ensemble summary
"""
# Print comprehensive ensemble summary
em.print_summary()
"""

# Cell 59: Plot alpha comparison
"""
# Plot performance vs alpha (similar to LangCell approach)
fig1 = em.plot_alpha_comparison(alphas=[0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
plt.show()
"""

# Cell 60: Confusion matrix for best alpha
"""
# Find best alpha and plot confusion matrix
best_alpha = max(results.keys(), key=lambda x: results[x]['f1_score'])
print(f"Best alpha: {best_alpha}")

fig2 = em.plot_confusion_matrix(alpha=best_alpha, method='embedding_ranking')
plt.show()
"""

# Cell 61: Save ensemble results
"""
# Save ensemble results to CSV
em.save_results(filename='geneformer_ensemble_metrics.csv')
print("Ensemble results saved to CSV file!")
"""

# Cell 62: Compare different methods
"""
# Compare different ensemble methods
print("Comparing different ensemble methods...")

methods = ['embedding_ranking', 'ranking_similarity', 'embedding_only']
method_results = {}

for method in methods:
    print(f"\nEvaluating {method}...")
    try:
        result = em.evaluate_ensemble(alpha=0.5, method=method)
        method_results[method] = result
        print(f"  F1-Score: {result['f1_score']:.3f}")
        print(f"  AUC-ROC: {result['auc_roc']:.3f}")
        print(f"  Accuracy: {result['accuracy']:.3f}")
    except Exception as e:
        print(f"  Error: {e}")

print(f"\nMethod comparison:")
for method, result in method_results.items():
    print(f"  {method}: F1={result['f1_score']:.3f}, AUC-ROC={result['auc_roc']:.3f}")
"""

# Cell 63: Detailed analysis for best configuration
"""
# Detailed analysis for best configuration
best_key = max(em.results.keys(), key=lambda x: em.results[x]['f1_score'])
best_results = em.results[best_key]

print(f"\nBest Configuration: {best_key}")
print(f"Method: {best_results['method']}")
print(f"Alpha: {best_results['alpha']}")
print(f"Accuracy: {best_results['accuracy']:.3f}")
print(f"Precision: {best_results['precision']:.3f}")
print(f"Recall: {best_results['recall']:.3f}")
print(f"F1-Score: {best_results['f1_score']:.3f}")
print(f"AUC-ROC: {best_results['auc_roc']:.3f}")
print(f"Avg Precision: {best_results['avg_precision']:.3f}")

# Per-class analysis
print(f"\nPer-class F1 scores:")
for class_name, f1_score in best_results['per_class_f1'].items():
    print(f"  {class_name}: {f1_score:.3f}")

print(f"\nPer-class AUC-ROC scores:")
for class_name, auc_roc in best_results['per_class_auc_roc'].items():
    print(f"  {class_name}: {auc_roc:.3f}")

print(f"\nPer-class Average Precision scores:")
for class_name, avg_precision in best_results['per_class_avg_precision'].items():
    print(f"  {class_name}: {avg_precision:.3f}")
"""

# Cell 64: Embedding vs Ranking analysis
"""
# Analyze contribution of embeddings vs rankings
print("Analyzing contribution of embeddings vs rankings...")

embedding_only = em.evaluate_ensemble(alpha=1.0, method='embedding_ranking')
ranking_only = em.evaluate_ensemble(alpha=0.0, method='embedding_ranking')
combined = em.evaluate_ensemble(alpha=0.5, method='embedding_ranking')

print(f"\nEmbedding-only (α=1.0):")
print(f"  F1-Score: {embedding_only['f1_score']:.3f}")
print(f"  AUC-ROC: {embedding_only['auc_roc']:.3f}")

print(f"\nRanking-only (α=0.0):")
print(f"  F1-Score: {ranking_only['f1_score']:.3f}")
print(f"  AUC-ROC: {ranking_only['auc_roc']:.3f}")

print(f"\nCombined (α=0.5):")
print(f"  F1-Score: {combined['f1_score']:.3f}")
print(f"  AUC-ROC: {combined['auc_roc']:.3f}")

# Calculate improvement
embedding_f1 = embedding_only['f1_score']
ranking_f1 = ranking_only['f1_score']
combined_f1 = combined['f1_score']

print(f"\nImprovement Analysis:")
print(f"  Best individual: {max(embedding_f1, ranking_f1):.3f}")
print(f"  Combined: {combined_f1:.3f}")
print(f"  Improvement: {combined_f1 - max(embedding_f1, ranking_f1):.3f}")
"""

# Cell 65: Final ensemble summary
"""
# Summary of ensemble evaluation
print("\n" + "="*80)
print("GENEFORMER ENSEMBLE EVALUATION SUMMARY (LangCell-style)")
print("="*80)

print(f"Dataset: {dataset_name}")
print(f"Model: Geneformer-V2-104M_CLcancer")
print(f"Total cells: {len(em.embeddings)}")
print(f"Cell types: {len(em.class_names)}")
print(f"Embedding dimension: {em.embeddings.shape[1]}")

if em.results:
    best_key = max(em.results.keys(), key=lambda x: em.results[x]['f1_score'])
    best_results = em.results[best_key]
    
    print(f"\nBest Configuration: {best_key}")
    print(f"Method: {best_results['method']}")
    print(f"Alpha: {best_results['alpha']}")
    print(f"Best F1-Score: {best_results['f1_score']:.3f}")
    print(f"Best Accuracy: {best_results['accuracy']:.3f}")
    print(f"Best Precision: {best_results['precision']:.3f}")
    print(f"Best Recall: {best_results['recall']:.3f}")
    print(f"Best AUC-ROC: {best_results['auc_roc']:.3f}")
    print(f"Best Avg Precision: {best_results['avg_precision']:.3f}")

print("\nResults saved to:")
print(f"  - CSV: {model_out}/geneformer_ensemble_metrics.csv")
print(f"  - Plots: {model_out}/alpha_comparison_*.png")
print(f"  - Plots: {model_out}/confusion_matrix_*.png")

print("="*80)
""" 
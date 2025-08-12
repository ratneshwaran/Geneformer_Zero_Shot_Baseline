# Simplified Geneformer Ensemble Evaluation (Embeddings Only)
# This version focuses on embeddings to avoid ranking data structure issues

"""
# Simplified Geneformer Ensemble Evaluation (Embeddings Only)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    average_precision_score
)
from sklearn.preprocessing import LabelEncoder
from sc_foundation_evals import geneformer_ensemble_metrics

print("="*80)
print("SIMPLIFIED GENEFORMER ENSEMBLE EVALUATION (Embeddings Only)")
print("="*80)

# Create ensemble evaluation object
em = geneformer_ensemble_metrics.create_geneformer_ensemble_evaluation(
    geneform_instance=geneform,
    input_data=input_data,
    label_col=label_col,  # Your cell type column name
    output_dir=model_out  # Your output directory
)

print(f"\nEnsemble evaluation object created!")
print(f"Number of cells: {len(em.embeddings)}")
print(f"Embedding dimension: {em.embeddings.shape[1]}")
print(f"Number of cell types: {len(em.class_names)}")

# Evaluate with embedding-only method (should work reliably)
print(f"\nEvaluating with embedding-only method...")
try:
    embedding_result = em.evaluate_ensemble(alpha=1.0, method='embedding_only')
    print(f"Embedding-only evaluation successful!")
    print(f"  F1-Score: {embedding_result['f1_score']:.3f}")
    print(f"  AUC-ROC: {embedding_result['auc_roc']:.3f}")
    print(f"  Accuracy: {embedding_result['accuracy']:.3f}")
    print(f"  Precision: {embedding_result['precision']:.3f}")
    print(f"  Recall: {embedding_result['recall']:.3f}")
except Exception as e:
    print(f"Embedding-only evaluation failed: {e}")
    import traceback
    traceback.print_exc()

# Try different alpha values with embedding-only method
print(f"\nEvaluating different alpha values with embedding-only method...")
alphas = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
results = {}

for alpha in alphas:
    try:
        result = em.evaluate_ensemble(alpha=alpha, method='embedding_only')
        results[alpha] = result
        print(f"  Alpha {alpha}: F1={result['f1_score']:.3f}, AUC-ROC={result['auc_roc']:.3f}")
    except Exception as e:
        print(f"  Alpha {alpha}: Failed - {e}")

if results:
    # Find best alpha
    best_alpha = max(results.keys(), key=lambda x: results[x]['f1_score'])
    best_result = results[best_alpha]
    
    print(f"\nBest Configuration:")
    print(f"  Alpha: {best_alpha}")
    print(f"  F1-Score: {best_result['f1_score']:.3f}")
    print(f"  AUC-ROC: {best_result['auc_roc']:.3f}")
    print(f"  Accuracy: {best_result['accuracy']:.3f}")
    
    # Per-class analysis
    print(f"\nPer-class F1 scores:")
    for class_name, f1_score in best_result['per_class_f1'].items():
        print(f"  {class_name}: {f1_score:.3f}")
    
    print(f"\nPer-class AUC-ROC scores:")
    for class_name, auc_roc in best_result['per_class_auc_roc'].items():
        print(f"  {class_name}: {auc_roc:.3f}")
    
    # Create simple comparison plot
    print(f"\nCreating performance comparison plot...")
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        alphas_list = list(results.keys())
        f1_scores = [results[alpha]['f1_score'] for alpha in alphas_list]
        auc_scores = [results[alpha]['auc_roc'] for alpha in alphas_list]
        
        ax.plot(alphas_list, f1_scores, 'o-', label='F1-Score', linewidth=2, markersize=8)
        ax.plot(alphas_list, auc_scores, 's-', label='AUC-ROC', linewidth=2, markersize=8)
        
        ax.set_xlabel('Alpha (Weight for Embeddings)')
        ax.set_ylabel('Score')
        ax.set_title('Geneformer Embedding Performance')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Confusion matrix for best alpha
        print(f"\nCreating confusion matrix for best alpha ({best_alpha})...")
        try:
            predictions, _ = em.create_ensemble_predictions(alpha=best_alpha, method='embedding_only')
            cm = confusion_matrix(em.labels_encoded, predictions)
            
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=em.class_names, 
                       yticklabels=em.class_names, ax=ax)
            ax.set_title(f'Confusion Matrix - Embedding Only (α={best_alpha})')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Could not create confusion matrix: {e}")
            
    except Exception as e:
        print(f"Could not create performance plot: {e}")
    
    # Save results
    print(f"\nSaving results...")
    try:
        summary_data = []
        for alpha, result in results.items():
            row = {
                'Alpha': alpha,
                'Method': 'embedding_only',
                'Accuracy': result['accuracy'],
                'Precision': result['precision'],
                'Recall': result['recall'],
                'F1_Score': result['f1_score'],
                'AUC_ROC': result['auc_roc'],
                'Avg_Precision': result['avg_precision']
            }
            summary_data.append(row)
        
        df = pd.DataFrame(summary_data)
        output_path = os.path.join(model_out, 'geneformer_embedding_metrics.csv')
        df.to_csv(output_path, index=False)
        print(f"Results saved to: {output_path}")
    except Exception as e:
        print(f"Could not save results: {e}")
    
    # Final summary
    print(f"\n" + "="*80)
    print("FINAL EMBEDDING-ONLY EVALUATION SUMMARY")
    print("="*80)
    print(f"Dataset: {dataset_name}")
    print(f"Model: Geneformer-V2-104M_CLcancer")
    print(f"Total cells: {len(em.embeddings)}")
    print(f"Cell types: {len(em.class_names)}")
    print(f"Embedding dimension: {em.embeddings.shape[1]}")
    print(f"Best Alpha: {best_alpha}")
    print(f"Best F1-Score: {best_result['f1_score']:.3f}")
    print(f"Best AUC-ROC: {best_result['auc_roc']:.3f}")
    print(f"Best Accuracy: {best_result['accuracy']:.3f}")
    print("="*80)
    print("EMBEDDING-ONLY EVALUATION COMPLETE! 🚀")
    print("="*80)
else:
    print("No successful evaluations. Check the error messages above.")
""" 
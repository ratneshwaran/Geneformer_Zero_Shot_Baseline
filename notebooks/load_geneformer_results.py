#!/usr/bin/env python3
"""
Utility script to load and compare Geneformer results with LangCell results.
This script helps in loading saved results and performing comparisons.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    average_precision_score
)

def load_geneformer_results(results_path):
    """
    Load Geneformer results from saved file.
    
    Args:
        results_path (str): Path to the saved results .pt file
        
    Returns:
        dict: Dictionary containing the loaded results
    """
    results = torch.load(results_path, map_location='cpu')
    return results

def load_langcell_results(results_path):
    """
    Load LangCell results from saved file.
    
    Args:
        results_path (str): Path to the saved results .pt file
        
    Returns:
        dict: Dictionary containing the loaded results
    """
    results = torch.load(results_path, map_location='cpu')
    return results

def compare_results(geneformer_results, langcell_results, output_dir=None):
    """
    Compare Geneformer and LangCell results.
    
    Args:
        geneformer_results (dict): Geneformer results
        langcell_results (dict): LangCell results
        output_dir (str): Directory to save comparison plots
        
    Returns:
        dict: Comparison metrics
    """
    # Extract predictions and labels
    gf_preds = geneformer_results['preds'].numpy()
    gf_labels = geneformer_results['labels'].numpy()
    gf_scores = geneformer_results['logits'].numpy()
    
    lc_preds = langcell_results['preds'].numpy()
    lc_labels = langcell_results['labels'].numpy()
    lc_scores = langcell_results['logits'].numpy()
    
    # Calculate metrics for Geneformer
    gf_accuracy = accuracy_score(gf_labels, gf_preds)
    gf_f1 = f1_score(gf_labels, gf_preds, average='macro')
    gf_auc = roc_auc_score(gf_labels, gf_scores, multi_class='ovr', average='macro')
    
    # Calculate metrics for LangCell
    lc_accuracy = accuracy_score(lc_labels, lc_preds)
    lc_f1 = f1_score(lc_labels, lc_preds, average='macro')
    lc_auc = roc_auc_score(lc_labels, lc_scores, multi_class='ovr', average='macro')
    
    # Create comparison summary
    comparison = {
        'Geneformer': {
            'Accuracy': gf_accuracy,
            'F1_Score': gf_f1,
            'AUC_ROC': gf_auc
        },
        'LangCell': {
            'Accuracy': lc_accuracy,
            'F1_Score': lc_f1,
            'AUC_ROC': lc_auc
        }
    }
    
    # Print comparison
    print("="*60)
    print("COMPARISON: Geneformer vs LangCell")
    print("="*60)
    print(f"{'Metric':<15} {'Geneformer':<12} {'LangCell':<12} {'Difference':<12}")
    print("-"*60)
    print(f"{'Accuracy':<15} {gf_accuracy:<12.3f} {lc_accuracy:<12.3f} {gf_accuracy-lc_accuracy:<12.3f}")
    print(f"{'F1-Score':<15} {gf_f1:<12.3f} {lc_f1:<12.3f} {gf_f1-lc_f1:<12.3f}")
    print(f"{'AUC-ROC':<15} {gf_auc:<12.3f} {lc_auc:<12.3f} {gf_auc-lc_auc:<12.3f}")
    print("="*60)
    
    # Create comparison plot if output_dir is provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Create comparison bar plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        metrics = ['Accuracy', 'F1_Score', 'AUC_ROC']
        x = np.arange(len(metrics))
        width = 0.35
        
        gf_values = [comparison['Geneformer'][m] for m in metrics]
        lc_values = [comparison['LangCell'][m] for m in metrics]
        
        ax.bar(x - width/2, gf_values, width, label='Geneformer', alpha=0.8)
        ax.bar(x + width/2, lc_values, width, label='LangCell', alpha=0.8)
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Score')
        ax.set_title('Geneformer vs LangCell Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'geneformer_vs_langcell_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save comparison metrics
        comparison_df = pd.DataFrame(comparison)
        comparison_df.to_csv(output_dir / 'geneformer_vs_langcell_metrics.csv')
        
        print(f"Comparison plots and metrics saved to: {output_dir}")
    
    return comparison

def analyze_per_class_performance(geneformer_results, langcell_results, output_dir=None):
    """
    Analyze per-class performance for both models.
    
    Args:
        geneformer_results (dict): Geneformer results
        langcell_results (dict): LangCell results
        output_dir (str): Directory to save analysis plots
    """
    # Extract data
    gf_preds = geneformer_results['preds'].numpy()
    gf_labels = geneformer_results['labels'].numpy()
    gf_class_names = geneformer_results['class_names']
    
    lc_preds = langcell_results['preds'].numpy()
    lc_labels = langcell_results['labels'].numpy()
    lc_class_names = langcell_results['class_names']
    
    # Calculate per-class F1 scores
    gf_f1_per_class = f1_score(gf_labels, gf_preds, average=None)
    lc_f1_per_class = f1_score(lc_labels, lc_preds, average=None)
    
    # Create per-class comparison
    per_class_data = []
    for i, class_name in enumerate(gf_class_names):
        per_class_data.append({
            'Class': class_name,
            'Geneformer_F1': gf_f1_per_class[i],
            'LangCell_F1': lc_f1_per_class[i] if i < len(lc_f1_per_class) else np.nan
        })
    
    per_class_df = pd.DataFrame(per_class_data)
    
    # Print per-class comparison
    print("\n" + "="*80)
    print("PER-CLASS PERFORMANCE COMPARISON")
    print("="*80)
    print(f"{'Class':<30} {'Geneformer_F1':<15} {'LangCell_F1':<15} {'Difference':<15}")
    print("-"*80)
    for _, row in per_class_df.iterrows():
        diff = row['Geneformer_F1'] - row['LangCell_F1']
        print(f"{row['Class']:<30} {row['Geneformer_F1']:<15.3f} {row['LangCell_F1']:<15.3f} {diff:<15.3f}")
    
    # Create per-class comparison plot if output_dir is provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        fig, ax = plt.subplots(figsize=(15, 8))
        
        x = np.arange(len(per_class_df))
        width = 0.35
        
        ax.bar(x - width/2, per_class_df['Geneformer_F1'], width, label='Geneformer', alpha=0.8)
        ax.bar(x + width/2, per_class_df['LangCell_F1'], width, label='LangCell', alpha=0.8)
        
        ax.set_xlabel('Cell Types')
        ax.set_ylabel('F1 Score')
        ax.set_title('Per-Class F1 Score Comparison: Geneformer vs LangCell')
        ax.set_xticks(x)
        ax.set_xticklabels(per_class_df['Class'], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'per_class_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save per-class metrics
        per_class_df.to_csv(output_dir / 'per_class_metrics.csv', index=False)
        
        print(f"Per-class analysis saved to: {output_dir}")

def main():
    """
    Main function to demonstrate usage.
    """
    print("Geneformer vs LangCell Results Comparison Utility")
    print("="*50)
    
    # Example usage
    geneformer_path = "../output/geneformer/pancreas_baseline/model_outputs/geneformer_pancreas_results.pt"
    langcell_path = "../LangCell/your_data/results_pancreas.pt"
    output_dir = "../output/comparisons/"
    
    # Check if files exist
    if not Path(geneformer_path).exists():
        print(f"Geneformer results not found at: {geneformer_path}")
        print("Please run the Geneformer baseline notebook first.")
        return
    
    if not Path(langcell_path).exists():
        print(f"LangCell results not found at: {langcell_path}")
        print("Please run the LangCell notebook first.")
        return
    
    # Load results
    print("Loading Geneformer results...")
    geneformer_results = load_geneformer_results(geneformer_path)
    
    print("Loading LangCell results...")
    langcell_results = load_langcell_results(langcell_path)
    
    # Compare results
    comparison = compare_results(geneformer_results, langcell_results, output_dir)
    
    # Analyze per-class performance
    analyze_per_class_performance(geneformer_results, langcell_results, output_dir)
    
    print("\nComparison complete!")

if __name__ == "__main__":
    main() 
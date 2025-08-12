## Copyright (c) Microsoft Corporation.
## Licensed under the MIT license.
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    precision_recall_curve, roc_curve, average_precision_score
)
from sklearn.preprocessing import LabelEncoder
import torch
import warnings
warnings.filterwarnings('ignore')

from .helpers.custom_logging import log

class GeneformerEnsembleMetrics:
    """
    Evaluate Geneformer using ensemble predictions similar to LangCell approach.
    Combines different model outputs rather than training separate classifiers.
    """
    
    def __init__(self, 
                 geneform_instance,
                 input_data,
                 label_col: str = "cell_type",
                 output_dir: Optional[str] = None):
        """
        Initialize ensemble metrics calculator.
        
        Parameters:
        -----------
        geneform_instance
            Geneformer instance with extracted embeddings and rankings
        input_data
            Input data object
        label_col : str
            Column name for cell type labels
        output_dir : Optional[str]
            Directory to save plots and results
        """
        self.geneform = geneform_instance
        self.input_data = input_data
        self.label_col = label_col
        self.output_dir = output_dir
        
        # Extract embeddings and rankings
        self.embeddings = geneform_instance.cell_embeddings
        self.input_rankings = geneform_instance.input_rankings
        self.output_rankings = geneform_instance.output_rankings
        self.labels = input_data.adata.obs[label_col].values
        
        # Encode labels to numeric
        self.label_encoder = LabelEncoder()
        self.labels_encoded = self.label_encoder.fit_transform(self.labels)
        self.class_names = self.label_encoder.classes_
        
        # Initialize results storage
        self.results = {}
        
        log.info(f"Initialized GeneformerEnsembleMetrics with {len(self.embeddings)} cells and {len(self.class_names)} classes")
        log.info(f"Classes: {list(self.class_names)}")
        log.info(f"Embedding shape: {self.embeddings.shape}")
        log.info(f"Input rankings length: {len(self.input_rankings)}")
        log.info(f"Output rankings length: {len(self.output_rankings)}")
    
    def create_ensemble_predictions(self, 
                                   alpha: float = 0.5,
                                   method: str = 'embedding_ranking') -> np.ndarray:
        """
        Create ensemble predictions by combining different model outputs.
        
        Parameters:
        -----------
        alpha : float
            Weight for combining different predictions (0-1)
        method : str
            Method for creating predictions ('embedding_ranking', 'ranking_similarity', 'embedding_only')
            
        Returns:
        --------
        np.ndarray
            Ensemble predictions
        """
        if method == 'embedding_ranking':
            # Combine embeddings with ranking similarity
            embedding_scores = self._get_embedding_scores()
            try:
                ranking_scores = self._get_ranking_similarity_scores()
                # Combine with alpha weight
                ensemble_scores = alpha * embedding_scores + (1 - alpha) * ranking_scores
            except Exception as e:
                log.warning(f"Ranking similarity failed, using embedding-only: {e}")
                ensemble_scores = embedding_scores
            
        elif method == 'ranking_similarity':
            # Use only ranking similarity
            try:
                ensemble_scores = self._get_ranking_similarity_scores()
            except Exception as e:
                log.warning(f"Ranking similarity failed, falling back to embedding-only: {e}")
                ensemble_scores = self._get_embedding_scores()
            
        elif method == 'embedding_only':
            # Use only embeddings
            ensemble_scores = self._get_embedding_scores()
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Convert to predictions
        predictions = np.argmax(ensemble_scores, axis=1)
        
        return predictions, ensemble_scores
    
    def _get_embedding_scores(self) -> np.ndarray:
        """
        Get prediction scores from embeddings using cosine similarity.
        """
        # Normalize embeddings
        embeddings_norm = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        
        # Calculate cosine similarity with class centroids
        scores = np.zeros((len(self.embeddings), len(self.class_names)))
        
        for i, class_name in enumerate(self.class_names):
            # Get embeddings for this class
            class_mask = self.labels == class_name
            if np.sum(class_mask) > 0:
                class_embeddings = embeddings_norm[class_mask]
                class_centroid = np.mean(class_embeddings, axis=0)
                class_centroid = class_centroid / np.linalg.norm(class_centroid)
                
                # Calculate similarity to centroid
                scores[:, i] = np.dot(embeddings_norm, class_centroid)
        
        return scores
    
    def _get_ranking_similarity_scores(self) -> np.ndarray:
        """
        Get prediction scores from ranking similarity.
        """
        scores = np.zeros((len(self.input_rankings), len(self.class_names)))
        
        # Check if rankings have consistent shapes
        try:
            # Try to convert to numpy array to check consistency
            test_input = np.array(self.input_rankings[:10])  # Test first 10
            test_output = np.array(self.output_rankings[:10])
            log.info(f"Ranking shapes - Input: {test_input.shape}, Output: {test_output.shape}")
        except Exception as e:
            log.warning(f"Rankings have inconsistent shapes: {e}")
            log.warning("Using embedding-only method instead")
            return self._get_embedding_scores()  # Fallback to embeddings only
        
        for i, class_name in enumerate(self.class_names):
            # Get rankings for this class
            class_mask = self.labels == class_name
            if np.sum(class_mask) > 0:
                try:
                    class_input_rankings = [self.input_rankings[j] for j in range(len(self.input_rankings)) if class_mask[j]]
                    class_output_rankings = [self.output_rankings[j] for j in range(len(self.output_rankings)) if class_mask[j]]
                    
                    # Convert to numpy arrays
                    class_input_array = np.array(class_input_rankings)
                    class_output_array = np.array(class_output_rankings)
                    
                    # Calculate average ranking for this class
                    avg_input_ranking = np.mean(class_input_array, axis=0)
                    avg_output_ranking = np.mean(class_output_array, axis=0)
                    
                    # Calculate similarity to average rankings
                    for j in range(len(self.input_rankings)):
                        try:
                            input_sim = self._calculate_ranking_similarity(self.input_rankings[j], avg_input_ranking)
                            output_sim = self._calculate_ranking_similarity(self.output_rankings[j], avg_output_ranking)
                            scores[j, i] = (input_sim + output_sim) / 2
                        except Exception as e:
                            log.warning(f"Error calculating similarity for cell {j}: {e}")
                            scores[j, i] = 0.0
                except Exception as e:
                    log.warning(f"Error processing class {class_name}: {e}")
                    # Set all scores for this class to 0
                    scores[:, i] = 0.0
        
        return scores
    
    def _calculate_ranking_similarity(self, ranking1: np.ndarray, ranking2: np.ndarray) -> float:
        """
        Calculate similarity between two rankings using Spearman correlation.
        """
        try:
            # Use Spearman correlation as similarity measure
            correlation = np.corrcoef(ranking1, ranking2)[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
        except:
            return 0.0
    
    def evaluate_ensemble(self, 
                         alpha: float = 0.5,
                         method: str = 'embedding_ranking') -> Dict[str, float]:
        """
        Evaluate ensemble predictions.
        
        Parameters:
        -----------
        alpha : float
            Weight for combining predictions
        method : str
            Method for creating predictions
            
        Returns:
        --------
        Dict[str, float]
            Performance metrics
        """
        # Get predictions
        predictions, scores = self.create_ensemble_predictions(alpha, method)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(self.labels_encoded, predictions),
            'precision': precision_score(self.labels_encoded, predictions, average='weighted', zero_division=0),
            'recall': recall_score(self.labels_encoded, predictions, average='weighted', zero_division=0),
            'f1_score': f1_score(self.labels_encoded, predictions, average='weighted', zero_division=0),
            'alpha': alpha,
            'method': method
        }
        
        # Calculate AUC-ROC and Average Precision for each class
        auc_roc_scores = []
        avg_precision_scores = []
        
        for i in range(len(self.class_names)):
            # One-vs-rest for each class
            y_true_binary = (self.labels_encoded == i).astype(int)
            y_scores_binary = scores[:, i]
            
            try:
                auc = roc_auc_score(y_true_binary, y_scores_binary)
                auc_roc_scores.append(auc)
            except:
                auc_roc_scores.append(0.0)
            
            try:
                ap = average_precision_score(y_true_binary, y_scores_binary)
                avg_precision_scores.append(ap)
            except:
                avg_precision_scores.append(0.0)
        
        # Average across classes
        metrics['auc_roc'] = np.mean(auc_roc_scores)
        metrics['avg_precision'] = np.mean(avg_precision_scores)
        
        # Per-class metrics
        per_class_precision = precision_score(self.labels_encoded, predictions, average=None, zero_division=0)
        per_class_recall = recall_score(self.labels_encoded, predictions, average=None, zero_division=0)
        per_class_f1 = f1_score(self.labels_encoded, predictions, average=None, zero_division=0)
        
        metrics['per_class_precision'] = dict(zip(self.class_names, per_class_precision))
        metrics['per_class_recall'] = dict(zip(self.class_names, per_class_recall))
        metrics['per_class_f1'] = dict(zip(self.class_names, per_class_f1))
        metrics['per_class_auc_roc'] = dict(zip(self.class_names, auc_roc_scores))
        metrics['per_class_avg_precision'] = dict(zip(self.class_names, avg_precision_scores))
        
        # Store results
        result_key = f"{method}_alpha_{alpha}"
        self.results[result_key] = metrics
        
        log.info(f"Evaluated {method} (α={alpha}): Accuracy={metrics['accuracy']:.3f}, F1={metrics['f1_score']:.3f}, AUC-ROC={metrics['auc_roc']:.3f}")
        
        return metrics
    
    def evaluate_multiple_alphas(self, 
                             alphas: List[float] = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
                             method: str = 'embedding_ranking') -> Dict[str, Dict[str, float]]:
        """
        Evaluate ensemble with multiple alpha values.
        
        Parameters:
        -----------
        alphas : List[float]
            List of alpha values to test
        method : str
            Method for creating predictions
            
        Returns:
        --------
        Dict[str, Dict[str, float]]
            Results for each alpha value
        """
        all_results = {}
        
        for alpha in alphas:
            try:
                results = self.evaluate_ensemble(alpha, method)
                all_results[alpha] = results
                log.info(f"Successfully evaluated alpha={alpha}: F1={results['f1_score']:.3f}")
            except Exception as e:
                log.warning(f"Failed to evaluate alpha={alpha}: {e}")
                continue
        
        log.info(f"Completed evaluation of {len(all_results)} alpha values")
        return all_results
    
    def plot_alpha_comparison(self, 
                             alphas: List[float] = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
                             method: str = 'embedding_ranking',
                             figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot performance comparison across different alpha values.
        
        Parameters:
        -----------
        alphas : List[float]
            List of alpha values to test
        method : str
            Method for creating predictions
        figsize : Tuple[int, int]
            Figure size
            
        Returns:
        --------
        plt.Figure
            Alpha comparison plot
        """
        # Evaluate all alphas
        results = self.evaluate_multiple_alphas(alphas, method)
        
        if not results:
            log.warning("No results available for plotting. Run evaluate_multiple_alphas first.")
            return None
        
        # Prepare data
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc', 'avg_precision']
        data = []
        
        for alpha, result in results.items():
            for metric in metrics:
                if metric in result:
                    data.append({
                        'Alpha': alpha,
                        'Metric': metric.replace('_', ' ').title(),
                        'Value': result[metric]
                    })
        
        if not data:
            log.warning("No data available for plotting.")
            return None
            
        df = pd.DataFrame(data)
        
        # Check if DataFrame is empty or has issues
        if df.empty:
            log.warning("DataFrame is empty. Cannot create plot.")
            return None
            
        # Verify columns exist
        required_columns = ['Alpha', 'Metric', 'Value']
        if not all(col in df.columns for col in required_columns):
            log.warning(f"Missing required columns. Available: {list(df.columns)}")
            return None
        
        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        try:
            sns.lineplot(data=df, x='Alpha', y='Value', hue='Metric', marker='o', ax=ax)
            ax.set_title(f'Performance vs Alpha - {method.replace("_", " ").title()}')
            ax.set_xlabel('Alpha (Weight for Embeddings)')
            ax.set_ylabel('Score')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            
            if self.output_dir:
                plt.savefig(os.path.join(self.output_dir, f'alpha_comparison_{method}.png'), 
                           dpi=300, bbox_inches='tight')
            
            return fig
        except Exception as e:
            log.error(f"Error creating plot: {e}")
            log.error(f"DataFrame shape: {df.shape}")
            log.error(f"DataFrame columns: {list(df.columns)}")
            log.error(f"DataFrame head:\n{df.head()}")
            return None
    
    def plot_confusion_matrix(self, 
                             alpha: float = 0.5,
                             method: str = 'embedding_ranking',
                             figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Plot confusion matrix for ensemble predictions.
        
        Parameters:
        -----------
        alpha : float
            Alpha value for ensemble
        method : str
            Method for creating predictions
        figsize : Tuple[int, int]
            Figure size
            
        Returns:
        --------
        plt.Figure
            Confusion matrix plot
        """
        # Get predictions
        predictions, _ = self.create_ensemble_predictions(alpha, method)
        
        # Create confusion matrix
        cm = confusion_matrix(self.labels_encoded, predictions)
        
        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, 
                   yticklabels=self.class_names, ax=ax)
        ax.set_title(f'Confusion Matrix - {method.replace("_", " ").title()} (α={alpha})')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        
        if self.output_dir:
            plt.savefig(os.path.join(self.output_dir, f'confusion_matrix_{method}_alpha_{alpha}.png'), 
                       dpi=300, bbox_inches='tight')
        
        return fig
    
    def print_summary(self) -> None:
        """
        Print a summary of all results.
        """
        if not self.results:
            log.warning("No results available. Run evaluate_ensemble first.")
            return
        
        print("\n" + "="*80)
        print("GENEFORMER ENSEMBLE EVALUATION SUMMARY")
        print("="*80)
        
        for result_key, results in self.results.items():
            print(f"\n{result_key}:")
            print(f"  Accuracy:      {results['accuracy']:.3f}")
            print(f"  Precision:     {results['precision']:.3f}")
            print(f"  Recall:        {results['recall']:.3f}")
            print(f"  F1-Score:      {results['f1_score']:.3f}")
            print(f"  AUC-ROC:       {results['auc_roc']:.3f}")
            print(f"  Avg Precision: {results['avg_precision']:.3f}")
        
        # Find best result
        best_key = max(self.results.keys(), 
                      key=lambda x: self.results[x]['f1_score'])
        best_results = self.results[best_key]
        
        print(f"\nBest Configuration: {best_key}")
        print(f"Best F1-Score: {best_results['f1_score']:.3f}")
        print(f"Best AUC-ROC: {best_results['auc_roc']:.3f}")
        print("="*80)
    
    def save_results(self, filename: str = 'geneformer_ensemble_metrics.csv') -> None:
        """
        Save results to CSV file.
        
        Parameters:
        -----------
        filename : str
            Output filename
        """
        if not self.output_dir:
            log.warning("No output directory specified. Results not saved.")
            return
        
        # Prepare summary data
        summary_data = []
        for result_key, results in self.results.items():
            row = {
                'Configuration': result_key,
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1_Score': results['f1_score'],
                'AUC_ROC': results['auc_roc'],
                'Avg_Precision': results['avg_precision']
            }
            summary_data.append(row)
        
        df = pd.DataFrame(summary_data)
        output_path = os.path.join(self.output_dir, filename)
        df.to_csv(output_path, index=False)
        log.info(f"Results saved to {output_path}")


def create_geneformer_ensemble_evaluation(geneform_instance,
                                        input_data,
                                        label_col: str = "cell_type",
                                        output_dir: Optional[str] = None) -> GeneformerEnsembleMetrics:
    """
    Create a Geneformer ensemble evaluation object.
    
    Parameters:
    -----------
    geneform_instance
        Geneformer instance with extracted embeddings and rankings
    input_data
        Input data object
    label_col : str
        Column name for cell type labels
    output_dir : Optional[str]
        Directory to save results
        
    Returns:
    --------
    GeneformerEnsembleMetrics
        Ensemble metrics object
    """
    # Create ensemble metrics object
    em = GeneformerEnsembleMetrics(geneform_instance, input_data, label_col, output_dir)
    
    return em 
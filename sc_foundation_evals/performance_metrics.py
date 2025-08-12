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
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

from .helpers.custom_logging import log

class PerformanceMetrics:
    """
    Comprehensive performance metrics for Geneformer evaluation.
    Provides standard classification metrics and visualization tools.
    """
    
    def __init__(self, 
                 embeddings: np.ndarray,
                 labels: np.ndarray,
                 label_key: str = "cell_type",
                 output_dir: Optional[str] = None):
        """
        Initialize performance metrics calculator.
        
        Parameters:
        -----------
        embeddings : np.ndarray
            Cell embeddings from Geneformer (n_cells x embedding_dim)
        labels : np.ndarray
            Cell type labels (n_cells,)
        label_key : str
            Name of the label column
        output_dir : Optional[str]
            Directory to save plots and results
        """
        self.embeddings = embeddings
        self.labels = labels
        self.label_key = label_key
        self.output_dir = output_dir
        
        # Encode labels to numeric
        self.label_encoder = LabelEncoder()
        self.labels_encoded = self.label_encoder.fit_transform(labels)
        self.class_names = self.label_encoder.classes_
        
        # Initialize results storage
        self.results = {}
        self.models = {}
        
        log.info(f"Initialized PerformanceMetrics with {len(embeddings)} cells and {len(self.class_names)} classes")
        log.info(f"Classes: {list(self.class_names)}")
    
    def calculate_basic_metrics(self, 
                               y_true: np.ndarray, 
                               y_pred: np.ndarray,
                               y_pred_proba: Optional[np.ndarray] = None,
                               average: str = 'weighted') -> Dict[str, float]:
        """
        Calculate basic classification metrics including AUC-ROC and Average Precision.
        
        Parameters:
        -----------
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted labels
        y_pred_proba : Optional[np.ndarray]
            Predicted probabilities (required for AUC-ROC and Average Precision)
        average : str
            Averaging method for multi-class metrics ('micro', 'macro', 'weighted')
            
        Returns:
        --------
        Dict[str, float]
            Dictionary containing accuracy, precision, recall, f1-score, auc_roc, avg_precision
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
            'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average=average, zero_division=0)
        }
        
        # Per-class metrics
        per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
        per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
        per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        metrics['per_class_precision'] = dict(zip(self.class_names, per_class_precision))
        metrics['per_class_recall'] = dict(zip(self.class_names, per_class_recall))
        metrics['per_class_f1'] = dict(zip(self.class_names, per_class_f1))
        
        # AUC-ROC and Average Precision (if probabilities are available)
        if y_pred_proba is not None:
            try:
                # Multi-class AUC-ROC (one-vs-rest)
                auc_roc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average=average)
                metrics['auc_roc'] = auc_roc
                
                # Average Precision (one-vs-rest)
                avg_precision = average_precision_score(y_true, y_pred_proba, average=average)
                metrics['avg_precision'] = avg_precision
                
                # Per-class AUC-ROC
                per_class_auc_roc = []
                per_class_avg_precision = []
                
                for i in range(len(self.class_names)):
                    # One-vs-rest for each class
                    y_true_binary = (y_true == i).astype(int)
                    y_pred_proba_binary = y_pred_proba[:, i]
                    
                    try:
                        auc = roc_auc_score(y_true_binary, y_pred_proba_binary)
                        per_class_auc_roc.append(auc)
                    except:
                        per_class_auc_roc.append(0.0)
                    
                    try:
                        ap = average_precision_score(y_true_binary, y_pred_proba_binary)
                        per_class_avg_precision.append(ap)
                    except:
                        per_class_avg_precision.append(0.0)
                
                metrics['per_class_auc_roc'] = dict(zip(self.class_names, per_class_auc_roc))
                metrics['per_class_avg_precision'] = dict(zip(self.class_names, per_class_avg_precision))
                
            except Exception as e:
                log.warning(f"Could not calculate AUC-ROC and Average Precision: {e}")
                metrics['auc_roc'] = None
                metrics['avg_precision'] = None
        else:
            metrics['auc_roc'] = None
            metrics['avg_precision'] = None
        
        return metrics
    
    def evaluate_with_classifier(self, 
                                classifier_type: str = 'svm',  # Changed default to SVM
                                test_size: float = 0.2,
                                random_state: int = 42) -> Dict[str, float]:
        """
        Evaluate embeddings using a classifier.
        
        Parameters:
        -----------
        classifier_type : str
            Type of classifier ('random_forest', 'svm', 'logistic')
        test_size : float
            Proportion of data for testing
        random_state : int
            Random seed for reproducibility
            
        Returns:
        --------
        Dict[str, float]
            Performance metrics
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.embeddings, self.labels_encoded, 
            test_size=test_size, random_state=random_state, stratify=self.labels_encoded
        )
        
        # Initialize classifier
        if classifier_type == 'random_forest':
            classifier = RandomForestClassifier(n_estimators=100, random_state=random_state)
        elif classifier_type == 'svm':
            classifier = SVC(probability=True, random_state=random_state)
        elif classifier_type == 'logistic':
            classifier = LogisticRegression(random_state=random_state, max_iter=1000)
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")
        
        # Train and predict
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        y_pred_proba = classifier.predict_proba(X_test) if hasattr(classifier, 'predict_proba') else None
        
        # Store model
        self.models[classifier_type] = classifier
        
        # Calculate metrics
        metrics = self.calculate_basic_metrics(y_test, y_pred, y_pred_proba)
        metrics['classifier_type'] = classifier_type
        metrics['test_size'] = test_size
        
        # Store results
        self.results[classifier_type] = metrics
        
        log.info(f"Evaluated with {classifier_type}: Accuracy={metrics['accuracy']:.3f}, F1={metrics['f1_score']:.3f}")
        
        return metrics
    
    def evaluate_all_classifiers(self, test_size: float = 0.2) -> Dict[str, Dict[str, float]]:
        """
        Evaluate embeddings with multiple classifiers.
        
        Parameters:
        -----------
        test_size : float
            Proportion of data for testing
            
        Returns:
        --------
        Dict[str, Dict[str, float]]
            Results for all classifiers
        """
        classifiers = ['random_forest', 'svm', 'logistic']
        all_results = {}
        
        for classifier in classifiers:
            try:
                results = self.evaluate_with_classifier(classifier, test_size)
                all_results[classifier] = results
            except Exception as e:
                log.warning(f"Failed to evaluate {classifier}: {e}")
                continue
        
        return all_results
    
    def plot_confusion_matrix(self, 
                             classifier_type: str = 'random_forest',
                             figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Plot confusion matrix for a classifier.
        
        Parameters:
        -----------
        classifier_type : str
            Type of classifier to use
        figsize : Tuple[int, int]
            Figure size
            
        Returns:
        --------
        plt.Figure
            Confusion matrix plot
        """
        if classifier_type not in self.models:
            raise ValueError(f"Classifier {classifier_type} not found. Run evaluate_with_classifier first.")
        
        # Get predictions
        classifier = self.models[classifier_type]
        y_pred = classifier.predict(self.embeddings)
        
        # Create confusion matrix
        cm = confusion_matrix(self.labels_encoded, y_pred)
        
        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, 
                   yticklabels=self.class_names, ax=ax)
        ax.set_title(f'Confusion Matrix - {classifier_type.replace("_", " ").title()}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        
        if self.output_dir:
            plt.savefig(os.path.join(self.output_dir, f'confusion_matrix_{classifier_type}.png'), 
                       dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_metrics_comparison(self, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot comparison of metrics across different classifiers.
        
        Parameters:
        -----------
        figsize : Tuple[int, int]
            Figure size
            
        Returns:
        --------
        plt.Figure
            Metrics comparison plot
        """
        if not self.results:
            raise ValueError("No results available. Run evaluate_all_classifiers first.")
        
        # Prepare data
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc', 'avg_precision']
        classifiers = list(self.results.keys())
        
        # Filter out None values for AUC-ROC and Average Precision
        available_metrics = []
        for metric in metrics:
            has_values = False
            for classifier in classifiers:
                if metric in self.results[classifier] and self.results[classifier][metric] is not None:
                    has_values = True
                    break
            if has_values:
                available_metrics.append(metric)
        
        data = []
        for classifier in classifiers:
            for metric in available_metrics:
                value = self.results[classifier].get(metric)
                if value is not None:
                    data.append({
                        'Classifier': classifier.replace('_', ' ').title(),
                        'Metric': metric.replace('_', ' ').title(),
                        'Value': value
                    })
        
        df = pd.DataFrame(data)
        
        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        sns.barplot(data=df, x='Classifier', y='Value', hue='Metric', ax=ax)
        ax.set_title('Performance Metrics Comparison')
        ax.set_ylabel('Score')
        ax.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        if self.output_dir:
            plt.savefig(os.path.join(self.output_dir, 'metrics_comparison.png'), 
                       dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_per_class_metrics(self, 
                              classifier_type: str = 'random_forest',
                              figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Plot per-class metrics for a classifier.
        
        Parameters:
        -----------
        classifier_type : str
            Type of classifier to use
        figsize : Tuple[int, int]
            Figure size
            
        Returns:
        --------
        plt.Figure
            Per-class metrics plot
        """
        if classifier_type not in self.results:
            raise ValueError(f"Classifier {classifier_type} not found. Run evaluate_with_classifier first.")
        
        results = self.results[classifier_type]
        
        # Prepare data
        metrics = ['per_class_precision', 'per_class_recall', 'per_class_f1', 'per_class_auc_roc', 'per_class_avg_precision']
        data = []
        
        for metric in metrics:
            metric_name = metric.replace('per_class_', '').title()
            if metric in results:
                for class_name, value in results[metric].items():
                    if value is not None:
                        data.append({
                            'Class': class_name,
                            'Metric': metric_name,
                            'Value': value
                        })
        
        df = pd.DataFrame(data)
        
        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        sns.barplot(data=df, x='Class', y='Value', hue='Metric', ax=ax)
        ax.set_title(f'Per-Class Metrics - {classifier_type.replace("_", " ").title()}')
        ax.set_ylabel('Score')
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        
        if self.output_dir:
            plt.savefig(os.path.join(self.output_dir, f'per_class_metrics_{classifier_type}.png'), 
                       dpi=300, bbox_inches='tight')
        
        return fig
    
    def save_results(self, filename: str = 'performance_metrics.csv') -> None:
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
        for classifier, results in self.results.items():
            row = {
                'Classifier': classifier,
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1_Score': results['f1_score']
            }
            
            # Add AUC-ROC and Average Precision if available
            if 'auc_roc' in results and results['auc_roc'] is not None:
                row['AUC_ROC'] = results['auc_roc']
            if 'avg_precision' in results and results['avg_precision'] is not None:
                row['Avg_Precision'] = results['avg_precision']
            
            summary_data.append(row)
        
        df = pd.DataFrame(summary_data)
        output_path = os.path.join(self.output_dir, filename)
        df.to_csv(output_path, index=False)
        log.info(f"Results saved to {output_path}")
    
    def get_best_classifier(self) -> Tuple[str, Dict[str, float]]:
        """
        Get the best performing classifier based on F1-score.
        
        Returns:
        --------
        Tuple[str, Dict[str, float]]
            Best classifier name and its results
        """
        if not self.results:
            raise ValueError("No results available. Run evaluate_all_classifiers first.")
        
        best_classifier = max(self.results.keys(), 
                            key=lambda x: self.results[x]['f1_score'])
        
        return best_classifier, self.results[best_classifier]
    
    def print_summary(self) -> None:
        """
        Print a summary of all results.
        """
        if not self.results:
            log.warning("No results available. Run evaluate_all_classifiers first.")
            return
        
        print("\n" + "="*60)
        print("PERFORMANCE METRICS SUMMARY")
        print("="*60)
        
        for classifier, results in self.results.items():
            print(f"\n{classifier.replace('_', ' ').title()}:")
            print(f"  Accuracy:      {results['accuracy']:.3f}")
            print(f"  Precision:     {results['precision']:.3f}")
            print(f"  Recall:        {results['recall']:.3f}")
            print(f"  F1-Score:      {results['f1_score']:.3f}")
            
            # Add AUC-ROC and Average Precision if available
            if 'auc_roc' in results and results['auc_roc'] is not None:
                print(f"  AUC-ROC:       {results['auc_roc']:.3f}")
            if 'avg_precision' in results and results['avg_precision'] is not None:
                print(f"  Avg Precision: {results['avg_precision']:.3f}")
        
        best_classifier, best_results = self.get_best_classifier()
        print(f"\nBest Classifier: {best_classifier.replace('_', ' ').title()}")
        print(f"Best F1-Score: {best_results['f1_score']:.3f}")
        print("="*60)
    
    def plot_roc_curves(self, 
                        classifier_type: str = 'random_forest',
                        figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot ROC curves for each class (one-vs-rest).
        
        Parameters:
        -----------
        classifier_type : str
            Type of classifier to use
        figsize : Tuple[int, int]
            Figure size
            
        Returns:
        --------
        plt.Figure
            ROC curves plot
        """
        if classifier_type not in self.models:
            raise ValueError(f"Classifier {classifier_type} not found. Run evaluate_with_classifier first.")
        
        # Get predictions
        classifier = self.models[classifier_type]
        y_pred_proba = classifier.predict_proba(self.embeddings)
        
        # Plot ROC curves for each class
        fig, ax = plt.subplots(figsize=figsize)
        
        for i, class_name in enumerate(self.class_names):
            y_true_binary = (self.labels_encoded == i).astype(int)
            y_pred_proba_binary = y_pred_proba[:, i]
            
            try:
                fpr, tpr, _ = roc_curve(y_true_binary, y_pred_proba_binary)
                auc = roc_auc_score(y_true_binary, y_pred_proba_binary)
                ax.plot(fpr, tpr, label=f'{class_name} (AUC = {auc:.3f})')
            except:
                continue
        
        ax.plot([0, 1], [0, 1], 'k--', label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curves - {classifier_type.replace("_", " ").title()}')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if self.output_dir:
            plt.savefig(os.path.join(self.output_dir, f'roc_curves_{classifier_type}.png'), 
                       dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_precision_recall_curves(self, 
                                    classifier_type: str = 'random_forest',
                                    figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot Precision-Recall curves for each class (one-vs-rest).
        
        Parameters:
        -----------
        classifier_type : str
            Type of classifier to use
        figsize : Tuple[int, int]
            Figure size
            
        Returns:
        --------
        plt.Figure
            Precision-Recall curves plot
        """
        if classifier_type not in self.models:
            raise ValueError(f"Classifier {classifier_type} not found. Run evaluate_with_classifier first.")
        
        # Get predictions
        classifier = self.models[classifier_type]
        y_pred_proba = classifier.predict_proba(self.embeddings)
        
        # Plot Precision-Recall curves for each class
        fig, ax = plt.subplots(figsize=figsize)
        
        for i, class_name in enumerate(self.class_names):
            y_true_binary = (self.labels_encoded == i).astype(int)
            y_pred_proba_binary = y_pred_proba[:, i]
            
            try:
                precision, recall, _ = precision_recall_curve(y_true_binary, y_pred_proba_binary)
                ap = average_precision_score(y_true_binary, y_pred_proba_binary)
                ax.plot(recall, precision, label=f'{class_name} (AP = {ap:.3f})')
            except:
                continue
        
        # Add baseline (random classifier)
        no_skill = len(self.labels_encoded[self.labels_encoded == 0]) / len(self.labels_encoded)
        ax.plot([0, 1], [no_skill, no_skill], 'k--', label='Random')
        
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(f'Precision-Recall Curves - {classifier_type.replace("_", " ").title()}')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if self.output_dir:
            plt.savefig(os.path.join(self.output_dir, f'precision_recall_curves_{classifier_type}.png'), 
                       dpi=300, bbox_inches='tight')
        
        return fig


def create_performance_evaluation(geneform_instance,
                                input_data,
                                label_col: str = "cell_type",
                                output_dir: Optional[str] = None) -> PerformanceMetrics:
    """
    Create a performance evaluation object for Geneformer embeddings.
    
    Parameters:
    -----------
    geneform_instance
        Geneformer instance with extracted embeddings
    input_data
        Input data object
    label_col : str
        Column name for cell type labels
    output_dir : Optional[str]
        Directory to save results
        
    Returns:
    --------
    PerformanceMetrics
        Performance metrics object
    """
    # Extract embeddings and labels
    embeddings = geneform_instance.cell_embeddings
    labels = input_data.adata.obs[label_col].values
    
    # Create performance metrics object
    pm = PerformanceMetrics(embeddings, labels, label_col, output_dir)
    
    return pm 
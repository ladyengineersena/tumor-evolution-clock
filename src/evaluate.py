"""
Evaluation pipeline for tumor evolution models.
"""

import argparse
import os
import pickle
import json
import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score,
    mean_absolute_error, mean_squared_error, r2_score,
    confusion_matrix, classification_report
)
from scipy.stats import kendalltau
from src.features import FeatureExtractor, create_target_labels
from src.models import TumorEvolutionPredictor


def load_data(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load data from CSV files."""
    mutations_df = pd.read_csv(os.path.join(data_dir, 'mutations.csv'))
    clones_df = pd.read_csv(os.path.join(data_dir, 'clones.csv'))
    patients_df = pd.read_csv(os.path.join(data_dir, 'patients.csv'))
    
    return mutations_df, clones_df, patients_df


def evaluate_classification(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None) -> dict:
    """
    Evaluate classification performance.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional)
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'n_samples': len(y_true),
        'n_positive': int(np.sum(y_true)),
        'n_negative': int(len(y_true) - np.sum(y_true))
    }
    
    if y_proba is not None and len(np.unique(y_true)) > 1:
        try:
            metrics['roc_auc'] = float(roc_auc_score(y_true, y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba))
            metrics['pr_auc'] = float(average_precision_score(y_true, y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba))
        except:
            metrics['roc_auc'] = 0.0
            metrics['pr_auc'] = 0.0
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        metrics['tn'] = int(cm[0, 0])
        metrics['fp'] = int(cm[0, 1])
        metrics['fn'] = int(cm[1, 0])
        metrics['tp'] = int(cm[1, 1])
        metrics['precision'] = float(cm[1, 1] / (cm[1, 1] + cm[0, 1])) if (cm[1, 1] + cm[0, 1]) > 0 else 0.0
        metrics['recall'] = float(cm[1, 1] / (cm[1, 1] + cm[1, 0])) if (cm[1, 1] + cm[1, 0]) > 0 else 0.0
        metrics['f1'] = 2 * metrics['precision'] * metrics['recall'] / (metrics['precision'] + metrics['recall']) if (metrics['precision'] + metrics['recall']) > 0 else 0.0
    
    return metrics


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Evaluate regression performance.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary of metrics
    """
    return {
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'mse': float(mean_squared_error(y_true, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'r2': float(r2_score(y_true, y_pred)),
        'n_samples': len(y_true)
    }


def evaluate_temporal_order(
    mutations_df: pd.DataFrame,
    predictions: pd.DataFrame
) -> dict:
    """
    Evaluate temporal ordering predictions (Kendall tau).
    
    Args:
        mutations_df: Original mutations dataframe
        predictions: Predictions with mutation ordering
        
    Returns:
        Dictionary of ordering metrics
    """
    metrics = {}
    
    # Group by patient
    for patient_id in mutations_df['patient_id'].unique():
        patient_mutations = mutations_df[mutations_df['patient_id'] == patient_id]
        patient_preds = predictions[predictions['patient_id'] == patient_id]
        
        # Get actual order (by emergence time)
        actual_order = patient_mutations.groupby('mutation_id')['emergence_time_days'].first().sort_values()
        actual_order_list = actual_order.index.tolist()
        
        # Get predicted order (by predicted emergence probability or time)
        if 'predicted_emerge_prob' in patient_preds.columns:
            predicted_order = patient_preds.groupby('mutation_id')['predicted_emerge_prob'].mean().sort_values(ascending=False)
        elif 'predicted_vaf_increase' in patient_preds.columns:
            predicted_order = patient_preds.groupby('mutation_id')['predicted_vaf_increase'].mean().sort_values(ascending=False)
        else:
            continue
        
        predicted_order_list = predicted_order.index.tolist()
        
        # Calculate Kendall tau (only for mutations present in both)
        common_mutations = set(actual_order_list) & set(predicted_order_list)
        if len(common_mutations) < 2:
            continue
        
        actual_ranks = [actual_order_list.index(m) for m in common_mutations]
        predicted_ranks = [predicted_order_list.index(m) for m in common_mutations]
        
        tau, p_value = kendalltau(actual_ranks, predicted_ranks)
        
        if 'kendall_tau' not in metrics:
            metrics['kendall_tau'] = []
            metrics['kendall_tau_p'] = []
        
        metrics['kendall_tau'].append(tau)
        metrics['kendall_tau_p'].append(p_value)
    
    if 'kendall_tau' in metrics:
        metrics['kendall_tau_mean'] = float(np.mean(metrics['kendall_tau']))
        metrics['kendall_tau_std'] = float(np.std(metrics['kendall_tau']))
        del metrics['kendall_tau']
        del metrics['kendall_tau_p']
    
    return metrics


def evaluate_model(
    model: TumorEvolutionPredictor,
    mutations_df: pd.DataFrame,
    clones_df: pd.DataFrame,
    patients_df: pd.DataFrame,
    target_col: str = 'target_emerge'
) -> dict:
    """
    Evaluate model on test data.
    
    Args:
        model: Trained model
        mutations_df: Mutations dataframe
        clones_df: Clones dataframe
        patients_df: Patients dataframe
        target_col: Target column name
        
    Returns:
        Dictionary of evaluation metrics
    """
    from src.train import prepare_training_data
    
    # Prepare test data
    print("Preparing test data...")
    features_df, targets_df = prepare_training_data(
        mutations_df, clones_df, patients_df, pathway_activation=None
    )
    
    if len(features_df) == 0:
        return {'error': 'No features extracted'}
    
    # Prepare features
    feature_cols = [col for col in features_df.columns 
                   if col not in ['patient_id', 'mutation_id', 'timepoint', 
                                 'target_emerge', 'target_vaf_increase']]
    
    X = features_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    y_true = features_df[target_col].values
    
    # Make predictions
    print("Making predictions...")
    y_pred = model.predict(X)
    
    # Evaluate
    if model.task == 'classification':
        y_proba = model.predict_proba(X)
        metrics = evaluate_classification(y_true, y_pred, y_proba)
        
        # Add top-k accuracy
        if y_proba.shape[1] > 1:
            top_k = min(3, len(np.unique(y_true)))
            top_k_indices = np.argsort(y_proba[:, 1])[-top_k:]
            top_k_accuracy = float(np.mean(y_true[top_k_indices]))
            metrics[f'top_{top_k}_accuracy'] = top_k_accuracy
    else:
        metrics = evaluate_regression(y_true, y_pred)
    
    # Temporal ordering evaluation
    print("Evaluating temporal ordering...")
    predictions_df = features_df[['patient_id', 'mutation_id']].copy()
    if model.task == 'classification' and y_proba is not None:
        predictions_df['predicted_emerge_prob'] = y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba.flatten()
    else:
        predictions_df['predicted_vaf_increase'] = y_pred
    
    ordering_metrics = evaluate_temporal_order(mutations_df, predictions_df)
    metrics.update(ordering_metrics)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate tumor evolution prediction model')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model file')
    parser.add_argument('--data', type=str, required=True,
                       help='Directory containing data files')
    parser.add_argument('--out', type=str, default='outputs/evaluation_results.json',
                       help='Output path for evaluation results')
    parser.add_argument('--target', type=str, default='target_emerge',
                       choices=['target_emerge', 'target_vaf_increase'],
                       help='Target variable')
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.model}...")
    with open(args.model, 'rb') as f:
        model = pickle.load(f)
    
    # Load data
    print("Loading data...")
    mutations_df, clones_df, patients_df = load_data(args.data)
    
    # Evaluate
    print("Evaluating model...")
    metrics = evaluate_model(model, mutations_df, clones_df, patients_df, args.target)
    
    # Save results
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Print results
    print("\nEvaluation Results:")
    print("=" * 50)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    print(f"\nResults saved to {args.out}")


if __name__ == '__main__':
    main()


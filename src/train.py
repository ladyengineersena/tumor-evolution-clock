"""
Training pipeline for tumor evolution models.
"""

import argparse
import os
import pickle
import json
import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.model_selection import train_test_split
from src.features import FeatureExtractor, create_target_labels
from src.models import TumorEvolutionPredictor


def load_data(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load data from CSV files.
    
    Args:
        data_dir: Directory containing data files
        
    Returns:
        Tuple of (mutations_df, clones_df, patients_df)
    """
    mutations_df = pd.read_csv(os.path.join(data_dir, 'mutations.csv'))
    clones_df = pd.read_csv(os.path.join(data_dir, 'clones.csv'))
    patients_df = pd.read_csv(os.path.join(data_dir, 'patients.csv'))
    
    return mutations_df, clones_df, patients_df


def prepare_training_data(
    mutations_df: pd.DataFrame,
    clones_df: pd.DataFrame,
    patients_df: pd.DataFrame,
    pathway_activation: dict = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare features and targets for training.
    
    Args:
        mutations_df: Mutations dataframe
        clones_df: Clones dataframe
        patients_df: Patients dataframe
        pathway_activation: Optional pathway activation dictionary
        
    Returns:
        Tuple of (features_df, targets_df)
    """
    extractor = FeatureExtractor()
    
    # Create target labels
    print("Creating target labels...")
    targets_df = create_target_labels(mutations_df, prediction_horizon=1)
    
    # Extract features
    print("Extracting features...")
    features_list = []
    
    for _, target_row in targets_df.iterrows():
        patient_id = target_row['patient_id']
        timepoint = target_row['timepoint']
        
        # Extract features for this patient at this timepoint
        patient_features = extractor.extract_patient_features(
            mutations_df,
            clones_df,
            patients_df,
            pathway_activation=pathway_activation,
            patient_id=patient_id,
            timepoint=timepoint
        )
        
        # Filter to the specific mutation
        mutation_features = patient_features[
            patient_features['mutation_id'] == target_row['mutation_id']
        ]
        
        if len(mutation_features) > 0:
            feature_row = mutation_features.iloc[0].to_dict()
            feature_row['target_emerge'] = target_row['target_emerge']
            feature_row['target_vaf_increase'] = target_row['target_vaf_increase']
            features_list.append(feature_row)
    
    features_df = pd.DataFrame(features_list)
    
    return features_df, targets_df


def train_model(
    features_df: pd.DataFrame,
    target_col: str = 'target_emerge',
    model_type: str = 'xgboost',
    task: str = 'classification',
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[TumorEvolutionPredictor, dict]:
    """
    Train the model.
    
    Args:
        features_df: DataFrame with features and targets
        target_col: Name of target column
        model_type: Type of model to train
        task: Task type ('classification' or 'regression')
        test_size: Proportion of data for testing
        random_state: Random seed
        
    Returns:
        Tuple of (trained_model, metrics_dict)
    """
    # Prepare features and targets
    feature_cols = [col for col in features_df.columns 
                   if col not in ['patient_id', 'mutation_id', 'timepoint', 
                                 'target_emerge', 'target_vaf_increase']]
    
    X = features_df[feature_cols].fillna(0)
    y = features_df[target_col].values
    
    # Remove any infinite values
    X = X.replace([np.inf, -np.inf], 0)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if task == 'classification' else None
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print(f"Feature count: {len(feature_cols)}")
    
    # Train model
    print(f"Training {model_type} model...")
    model = TumorEvolutionPredictor(model_type=model_type, task=task)
    model.fit(X_train, y_train, feature_names=feature_cols)
    
    # Evaluate
    train_score = model.model.score(
        model.scaler.transform(X_train), y_train
    )
    test_score = model.model.score(
        model.scaler.transform(X_test), y_test
    )
    
    metrics = {
        'train_score': float(train_score),
        'test_score': float(test_score),
        'n_features': len(feature_cols),
        'n_train': len(X_train),
        'n_test': len(X_test)
    }
    
    print(f"Train score: {train_score:.4f}")
    print(f"Test score: {test_score:.4f}")
    
    return model, metrics


def main():
    parser = argparse.ArgumentParser(description='Train tumor evolution prediction model')
    parser.add_argument('--data', type=str, required=True,
                       help='Directory containing data files')
    parser.add_argument('--out', type=str, default='outputs/model.pkl',
                       help='Output path for trained model')
    parser.add_argument('--model_type', type=str, default='xgboost',
                       choices=['xgboost', 'random_forest'],
                       help='Type of model to train')
    parser.add_argument('--task', type=str, default='classification',
                       choices=['classification', 'regression'],
                       help='Task type')
    parser.add_argument('--target', type=str, default='target_emerge',
                       choices=['target_emerge', 'target_vaf_increase'],
                       help='Target variable')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Proportion of data for testing')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    
    # Load data
    print("Loading data...")
    mutations_df, clones_df, patients_df = load_data(args.data)
    
    # Load pathway activation if available
    pathway_file = os.path.join(args.data, 'cohort_summary.json')
    pathway_activation = None
    if os.path.exists(pathway_file):
        # In real implementation, load from patient-specific files
        pass
    
    # Prepare training data
    features_df, targets_df = prepare_training_data(
        mutations_df, clones_df, patients_df, pathway_activation
    )
    
    if len(features_df) == 0:
        print("ERROR: No features extracted. Check data format.")
        return
    
    # Train model
    model, metrics = train_model(
        features_df,
        target_col=args.target,
        model_type=args.model_type,
        task=args.task,
        test_size=args.test_size
    )
    
    # Save model
    print(f"Saving model to {args.out}...")
    with open(args.out, 'wb') as f:
        pickle.dump(model, f)
    
    # Save metrics
    metrics_file = args.out.replace('.pkl', '_metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save feature importance
    importance_df = model.get_feature_importance()
    importance_file = args.out.replace('.pkl', '_feature_importance.csv')
    importance_df.to_csv(importance_file, index=False)
    print(f"Top 10 features:")
    print(importance_df.head(10))
    
    print("Training complete!")


if __name__ == '__main__':
    main()


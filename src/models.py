"""
Machine learning models for tumor evolution prediction.

Includes:
- Baseline XGBoost models
- Phylogenetic tree utilities
- Ensemble methods
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import networkx as nx


class TumorEvolutionPredictor:
    """Main model for predicting tumor evolution timelines."""
    
    def __init__(
        self,
        model_type: str = 'xgboost',
        task: str = 'classification',  # 'classification' or 'regression'
        **model_kwargs
    ):
        """
        Initialize predictor.
        
        Args:
            model_type: Type of model ('xgboost', 'random_forest')
            task: Prediction task type
            **model_kwargs: Additional model parameters
        """
        self.model_type = model_type
        self.task = task
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = None
        
        if model_type == 'xgboost':
            if task == 'classification':
                self.model = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    **model_kwargs
                )
            else:
                self.model = xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    **model_kwargs
                )
        elif model_type == 'random_forest':
            if task == 'classification':
                self.model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    **model_kwargs
                )
            else:
                self.model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    **model_kwargs
                )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def fit(self, X: pd.DataFrame, y: np.ndarray, feature_names: Optional[List[str]] = None):
        """
        Train the model.
        
        Args:
            X: Feature matrix
            y: Target labels
            feature_names: List of feature names
        """
        if feature_names is None:
            if isinstance(X, pd.DataFrame):
                self.feature_names = X.columns.tolist()
            else:
                self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        else:
            self.feature_names = feature_names
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities (for classification).
        
        Args:
            X: Feature matrix
            
        Returns:
            Class probabilities
        """
        if self.task != 'classification':
            raise ValueError("predict_proba only available for classification tasks")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance scores.
        
        Returns:
            DataFrame with feature importances
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'get_booster'):
            # XGBoost
            importances = self.model.get_booster().get_score(importance_type='gain')
            # Convert to array format
            if isinstance(importances, dict):
                importances = np.array([importances.get(f'f{i}', 0) for i in range(len(self.feature_names))])
        else:
            raise ValueError("Cannot extract feature importance from this model")
        
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)


class PhylogeneticTreeBuilder:
    """Utilities for building and analyzing phylogenetic trees."""
    
    @staticmethod
    def build_tree_from_clones(clone_data: pd.DataFrame, patient_id: str) -> nx.DiGraph:
        """
        Build phylogenetic tree from clone data.
        
        Args:
            clone_data: DataFrame with clone information
            patient_id: Patient identifier
            
        Returns:
            Directed graph representing phylogenetic tree
        """
        patient_clones = clone_data[clone_data['patient_id'] == patient_id]
        
        tree = nx.DiGraph()
        
        # Get unique clones
        clones = patient_clones['clone_id'].unique()
        
        # Add root
        tree.add_node(0, name='root', time=0.0)
        
        # Build tree from parent-child relationships
        for clone_id in clones:
            clone_info = patient_clones[patient_clones['clone_id'] == clone_id].iloc[0]
            parent_id = clone_info.get('parent_id', 0)
            emergence_time = clone_info.get('emergence_time_days', 0.0)
            
            if parent_id is None:
                parent_id = 0
            
            tree.add_node(clone_id, name=f'clone_{clone_id}', time=emergence_time)
            tree.add_edge(parent_id, clone_id)
        
        return tree
    
    @staticmethod
    def infer_temporal_order(tree: nx.DiGraph) -> List[int]:
        """
        Infer temporal order of clones from tree.
        
        Args:
            tree: Phylogenetic tree
            
        Returns:
            List of clone IDs in temporal order
        """
        # Topological sort gives temporal order
        try:
            order = list(nx.topological_sort(tree))
            # Remove root
            if 0 in order:
                order.remove(0)
            return order
        except nx.NetworkXError:
            # If not a DAG, use time attribute
            nodes_with_time = [(n, tree.nodes[n].get('time', 0)) for n in tree.nodes() if n != 0]
            nodes_with_time.sort(key=lambda x: x[1])
            return [n for n, _ in nodes_with_time]
    
    @staticmethod
    def predict_next_clones(
        tree: nx.DiGraph,
        current_clones: List[int],
        clone_vaf: Dict[int, float]
    ) -> List[Tuple[int, float]]:
        """
        Predict which clones are likely to emerge next.
        
        Args:
            tree: Phylogenetic tree
            current_clones: List of currently observed clone IDs
            clone_vaf: Dictionary mapping clone ID to current VAF
            
        Returns:
            List of (clone_id, probability) tuples
        """
        candidates = []
        
        for node in tree.nodes():
            if node == 0 or node in current_clones:
                continue
            
            # Check if parent is present
            parents = list(tree.predecessors(node))
            if parents and any(p in current_clones for p in parents):
                # Parent exists, this clone could emerge
                parent_vaf = max([clone_vaf.get(p, 0) for p in parents])
                probability = min(1.0, parent_vaf * 0.5)  # Simplified probability
                candidates.append((node, probability))
        
        # Sort by probability
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates


class EnsemblePredictor:
    """Ensemble of multiple predictors."""
    
    def __init__(self, predictors: List[TumorEvolutionPredictor]):
        """
        Initialize ensemble.
        
        Args:
            predictors: List of predictor models
        """
        self.predictors = predictors
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make ensemble predictions (average).
        
        Args:
            X: Feature matrix
            
        Returns:
            Averaged predictions
        """
        predictions = [p.predict(X) for p in self.predictors]
        return np.mean(predictions, axis=0)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make ensemble probability predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Averaged probabilities
        """
        probas = [p.predict_proba(X) for p in self.predictors]
        return np.mean(probas, axis=0)


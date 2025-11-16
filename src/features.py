"""
Feature engineering for tumor evolution prediction.

Extracts features from multi-sample genomics data including:
- VAF trajectories and trends
- CCF (Cancer Cell Fraction) dynamics
- Pathway-level aggregation
- Clonal growth rates
- Treatment features
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
from scipy.stats import linregress


class FeatureExtractor:
    """Extract features from multi-sample genomics data."""
    
    def __init__(self):
        pass
    
    def extract_vaf_trajectory_features(
        self,
        vaf_values: np.ndarray,
        timepoints: np.ndarray
    ) -> Dict[str, float]:
        """
        Extract features from VAF trajectory.
        
        Args:
            vaf_values: Array of VAF values over time
            timepoints: Array of time points (days)
            
        Returns:
            Dictionary of trajectory features
        """
        features = {}
        
        if len(vaf_values) < 2:
            return self._empty_trajectory_features()
        
        # Basic statistics
        features['vaf_mean'] = np.mean(vaf_values)
        features['vaf_std'] = np.std(vaf_values)
        features['vaf_max'] = np.max(vaf_values)
        features['vaf_min'] = np.min(vaf_values)
        features['vaf_range'] = features['vaf_max'] - features['vaf_min']
        
        # Trend features
        if len(vaf_values) >= 2:
            slope, intercept, r_value, p_value, std_err = linregress(timepoints, vaf_values)
            features['vaf_slope'] = slope
            features['vaf_intercept'] = intercept
            features['vaf_r_squared'] = r_value ** 2
            features['vaf_p_value'] = p_value
        else:
            features['vaf_slope'] = 0.0
            features['vaf_intercept'] = vaf_values[0]
            features['vaf_r_squared'] = 0.0
            features['vaf_p_value'] = 1.0
        
        # Change features
        if len(vaf_values) >= 2:
            features['vaf_delta'] = vaf_values[-1] - vaf_values[0]
            features['vaf_relative_change'] = (vaf_values[-1] - vaf_values[0]) / (vaf_values[0] + 1e-6)
        else:
            features['vaf_delta'] = 0.0
            features['vaf_relative_change'] = 0.0
        
        # Acceleration (second derivative approximation)
        if len(vaf_values) >= 3:
            first_diff = np.diff(vaf_values)
            second_diff = np.diff(first_diff)
            features['vaf_acceleration'] = np.mean(second_diff) if len(second_diff) > 0 else 0.0
        else:
            features['vaf_acceleration'] = 0.0
        
        # Presence/emergence
        features['vaf_present'] = 1.0 if np.any(vaf_values > 0.01) else 0.0
        features['vaf_emerged'] = 1.0 if vaf_values[0] < 0.01 and np.any(vaf_values > 0.01) else 0.0
        
        return features
    
    def _empty_trajectory_features(self) -> Dict[str, float]:
        """Return empty feature set."""
        return {
            'vaf_mean': 0.0, 'vaf_std': 0.0, 'vaf_max': 0.0, 'vaf_min': 0.0,
            'vaf_range': 0.0, 'vaf_slope': 0.0, 'vaf_intercept': 0.0,
            'vaf_r_squared': 0.0, 'vaf_p_value': 1.0, 'vaf_delta': 0.0,
            'vaf_relative_change': 0.0, 'vaf_acceleration': 0.0,
            'vaf_present': 0.0, 'vaf_emerged': 0.0
        }
    
    def extract_clonal_features(
        self,
        clone_data: pd.DataFrame,
        patient_id: str
    ) -> Dict[str, float]:
        """
        Extract clonal-level features.
        
        Args:
            clone_data: DataFrame with clone VAF trajectories
            patient_id: Patient identifier
            
        Returns:
            Dictionary of clonal features
        """
        patient_clones = clone_data[clone_data['patient_id'] == patient_id]
        
        features = {}
        
        # Clone count
        n_clones = patient_clones['clone_id'].nunique()
        features['n_clones'] = float(n_clones)
        
        # Clone size distribution
        if n_clones > 0:
            latest_timepoint = patient_clones['timepoint'].max()
            latest_clones = patient_clones[patient_clones['timepoint'] == latest_timepoint]
            clone_sizes = latest_clones.groupby('clone_id')['vaf'].mean()
            
            features['clone_size_mean'] = clone_sizes.mean()
            features['clone_size_std'] = clone_sizes.std()
            features['clone_size_max'] = clone_sizes.max()
            features['clone_size_min'] = clone_sizes.min()
            features['clone_diversity'] = stats.entropy(clone_sizes + 1e-6)
        else:
            features['clone_size_mean'] = 0.0
            features['clone_size_std'] = 0.0
            features['clone_size_max'] = 0.0
            features['clone_size_min'] = 0.0
            features['clone_diversity'] = 0.0
        
        # Growth rates (slope of VAF over time)
        growth_rates = []
        for clone_id in patient_clones['clone_id'].unique():
            clone_traj = patient_clones[patient_clones['clone_id'] == clone_id].sort_values('timepoint')
            if len(clone_traj) >= 2:
                slope, _, _, _, _ = linregress(clone_traj['timepoint_days'], clone_traj['vaf'])
                growth_rates.append(slope)
        
        if growth_rates:
            features['growth_rate_mean'] = np.mean(growth_rates)
            features['growth_rate_std'] = np.std(growth_rates)
            features['growth_rate_max'] = np.max(growth_rates)
        else:
            features['growth_rate_mean'] = 0.0
            features['growth_rate_std'] = 0.0
            features['growth_rate_max'] = 0.0
        
        return features
    
    def extract_pathway_features(
        self,
        mutation_data: pd.DataFrame,
        pathway_activation: Dict[str, List[int]],
        patient_id: str
    ) -> Dict[str, float]:
        """
        Extract pathway-level features.
        
        Args:
            mutation_data: DataFrame with mutation data
            pathway_activation: Dictionary mapping pathways to active clone IDs
            patient_id: Patient identifier
            
        Returns:
            Dictionary of pathway features
        """
        patient_mutations = mutation_data[mutation_data['patient_id'] == patient_id]
        features = {}
        
        # Pathway activation scores (simplified - based on gene names)
        pathway_genes = {
            'PI3K': ['PIK3CA', 'PIK3CB', 'PTEN'],
            'MAPK': ['KRAS', 'BRAF', 'NRAS', 'MAP2K1'],
            'p53': ['TP53'],
            'Cell_Cycle': ['CDKN2A', 'RB1', 'CCND1'],
            'DNA_Repair': ['BRCA1', 'BRCA2', 'ATM']
        }
        
        latest_timepoint = patient_mutations['timepoint'].max()
        latest_mutations = patient_mutations[patient_mutations['timepoint'] == latest_timepoint]
        
        for pathway, genes in pathway_genes.items():
            # Check if any mutations in pathway genes
            pathway_mutations = latest_mutations[latest_mutations['gene'].str.contains('|'.join(genes), case=False, na=False)]
            features[f'pathway_{pathway}_mutated'] = 1.0 if len(pathway_mutations) > 0 else 0.0
            features[f'pathway_{pathway}_vaf_sum'] = pathway_mutations['vaf'].sum() if len(pathway_mutations) > 0 else 0.0
            features[f'pathway_{pathway}_n_mutations'] = float(len(pathway_mutations))
        
        return features
    
    def extract_treatment_features(
        self,
        patient_metadata: pd.DataFrame,
        patient_id: str,
        current_timepoint: int
    ) -> Dict[str, float]:
        """
        Extract treatment-related features.
        
        Args:
            patient_metadata: DataFrame with patient metadata
            patient_id: Patient identifier
            current_timepoint: Current timepoint index
            
        Returns:
            Dictionary of treatment features
        """
        patient = patient_metadata[patient_metadata['patient_id'] == patient_id]
        
        if len(patient) == 0:
            return {
                'treatment_type_encoded': 0.0,
                'time_since_treatment_start': 0.0,
                'on_treatment': 0.0
            }
        
        patient = patient.iloc[0]
        
        features = {}
        
        # Treatment type encoding
        treatment_types = {'None': 0, 'Chemotherapy': 1, 'Targeted': 2, 'Immunotherapy': 3}
        features['treatment_type_encoded'] = float(treatment_types.get(patient.get('treatment_type', 'None'), 0))
        
        # Time since treatment start (simplified - would need actual timepoint days)
        treatment_start_day = patient.get('treatment_start_day', 0)
        # Assuming timepoints are roughly evenly spaced, estimate current day
        # This is simplified - in real data, use actual timepoint_days
        features['time_since_treatment_start'] = max(0.0, float(current_timepoint * 100 - treatment_start_day))
        features['on_treatment'] = 1.0 if features['time_since_treatment_start'] > 0 else 0.0
        
        return features
    
    def extract_patient_features(
        self,
        mutation_data: pd.DataFrame,
        clone_data: pd.DataFrame,
        patient_metadata: pd.DataFrame,
        pathway_activation: Optional[Dict] = None,
        patient_id: str = None,
        timepoint: int = None
    ) -> pd.DataFrame:
        """
        Extract all features for a patient at a given timepoint.
        
        Args:
            mutation_data: DataFrame with mutation data
            clone_data: DataFrame with clone data
            patient_metadata: DataFrame with patient metadata
            pathway_activation: Optional pathway activation dictionary
            patient_id: Patient identifier
            timepoint: Timepoint index (if None, uses latest)
            
        Returns:
            DataFrame with features for each mutation/clone
        """
        if patient_id is None:
            patient_ids = mutation_data['patient_id'].unique()
        else:
            patient_ids = [patient_id]
        
        all_features = []
        
        for pid in patient_ids:
            patient_mutations = mutation_data[mutation_data['patient_id'] == pid]
            
            if timepoint is None:
                available_timepoints = patient_mutations['timepoint'].unique()
                if len(available_timepoints) == 0:
                    continue
                # Use second-to-last timepoint for prediction (predicting next)
                timepoint = sorted(available_timepoints)[-2] if len(available_timepoints) > 1 else available_timepoints[0]
            
            # Get mutations up to current timepoint
            historical_mutations = patient_mutations[patient_mutations['timepoint'] <= timepoint]
            
            # Extract features for each mutation
            for mut_id in historical_mutations['mutation_id'].unique():
                mut_data = historical_mutations[historical_mutations['mutation_id'] == mut_id].sort_values('timepoint')
                
                vaf_trajectory = mut_data['vaf'].values
                timepoints = mut_data['timepoint_days'].values
                
                # VAF trajectory features
                vaf_features = self.extract_vaf_trajectory_features(vaf_trajectory, timepoints)
                
                # Clone features
                clone_features = self.extract_clonal_features(clone_data, pid)
                
                # Pathway features
                if pathway_activation:
                    pathway_features = self.extract_pathway_features(
                        mutation_data, pathway_activation, pid
                    )
                else:
                    pathway_features = {}
                
                # Treatment features
                treatment_features = self.extract_treatment_features(
                    patient_metadata, pid, timepoint
                )
                
                # Combine all features
                feature_row = {
                    'patient_id': pid,
                    'mutation_id': mut_id,
                    'timepoint': timepoint,
                    **vaf_features,
                    **clone_features,
                    **pathway_features,
                    **treatment_features
                }
                
                all_features.append(feature_row)
        
        return pd.DataFrame(all_features)


def create_target_labels(
    mutation_data: pd.DataFrame,
    prediction_horizon: int = 1
) -> pd.DataFrame:
    """
    Create target labels for prediction tasks.
    
    Args:
        mutation_data: DataFrame with mutation data
        prediction_horizon: Number of timepoints ahead to predict
        
    Returns:
        DataFrame with target labels
    """
    targets = []
    
    for patient_id in mutation_data['patient_id'].unique():
        patient_mutations = mutation_data[mutation_data['patient_id'] == patient_id].sort_values('timepoint')
        timepoints = sorted(patient_mutations['timepoint'].unique())
        
        for i in range(len(timepoints) - prediction_horizon):
            current_tp = timepoints[i]
            future_tp = timepoints[i + prediction_horizon]
            
            current_mutations = patient_mutations[patient_mutations['timepoint'] == current_tp]
            future_mutations = patient_mutations[patient_mutations['timepoint'] == future_tp]
            
            for mut_id in current_mutations['mutation_id'].unique():
                current_vaf = current_mutations[current_mutations['mutation_id'] == mut_id]['vaf'].values
                future_vaf = future_mutations[future_mutations['mutation_id'] == mut_id]['vaf'].values
                
                if len(current_vaf) > 0 and len(future_vaf) > 0:
                    # Target: will mutation emerge/expand?
                    will_emerge = 1.0 if (current_vaf[0] < 0.05 and future_vaf[0] > 0.05) else 0.0
                    vaf_increase = max(0.0, future_vaf[0] - current_vaf[0])
                    
                    targets.append({
                        'patient_id': patient_id,
                        'mutation_id': mut_id,
                        'timepoint': current_tp,
                        'target_emerge': will_emerge,
                        'target_vaf_increase': vaf_increase,
                        'future_timepoint': future_tp
                    })
    
    return pd.DataFrame(targets)


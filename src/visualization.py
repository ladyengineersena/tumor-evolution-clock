"""
Visualization tools for tumor evolution analysis.

Includes:
- Timeline plots
- Phylogenetic tree visualization
- VAF trajectory plots
- Feature importance plots
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List, Optional
import json


def plot_vaf_trajectory(
    mutation_data: pd.DataFrame,
    mutation_id: str,
    patient_id: str,
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plot VAF trajectory for a specific mutation.
    
    Args:
        mutation_data: DataFrame with mutation data
        mutation_id: Mutation identifier
        patient_id: Patient identifier
        ax: Optional matplotlib axes
        
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    
    patient_mutations = mutation_data[
        (mutation_data['patient_id'] == patient_id) &
        (mutation_data['mutation_id'] == mutation_id)
    ].sort_values('timepoint')
    
    if len(patient_mutations) == 0:
        return ax
    
    ax.plot(patient_mutations['timepoint_days'], patient_mutations['vaf'],
            marker='o', linewidth=2, markersize=8, label=f'{mutation_id}')
    ax.set_xlabel('Time (days)', fontsize=12)
    ax.set_ylabel('VAF', fontsize=12)
    ax.set_title(f'VAF Trajectory: {mutation_id} (Patient: {patient_id})', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim([0, 1])
    
    return ax


def plot_clone_timeline(
    clone_data: pd.DataFrame,
    patient_id: str,
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plot clone emergence timeline.
    
    Args:
        clone_data: DataFrame with clone data
        patient_id: Patient identifier
        ax: Optional matplotlib axes
        
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    patient_clones = clone_data[clone_data['patient_id'] == patient_id]
    
    # Get unique clones
    clones = patient_clones['clone_id'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(clones)))
    
    for i, clone_id in enumerate(clones):
        clone_traj = patient_clones[patient_clones['clone_id'] == clone_id].sort_values('timepoint')
        ax.plot(clone_traj['timepoint_days'], clone_traj['vaf'],
               marker='o', linewidth=2, markersize=6,
               label=f'Clone {clone_id}', color=colors[i])
    
    ax.set_xlabel('Time (days)', fontsize=12)
    ax.set_ylabel('VAF', fontsize=12)
    ax.set_title(f'Clone Evolution Timeline (Patient: {patient_id})', fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    return ax


def plot_phylogenetic_tree(
    tree: nx.DiGraph,
    ax: Optional[plt.Axes] = None,
    layout: str = 'hierarchical'
) -> plt.Axes:
    """
    Plot phylogenetic tree.
    
    Args:
        tree: NetworkX directed graph representing phylogenetic tree
        ax: Optional matplotlib axes
        layout: Layout type ('hierarchical' or 'spring')
        
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    if layout == 'hierarchical':
        # Use hierarchical layout based on time
        pos = {}
        for node in tree.nodes():
            time = tree.nodes[node].get('time', 0)
            # Group nodes by time level
            level = int(time * 2)  # Scale time to levels
            # Count nodes at this level
            nodes_at_level = [n for n in tree.nodes() if int(tree.nodes[n].get('time', 0) * 2) == level]
            idx = nodes_at_level.index(node)
            pos[node] = (level, idx - len(nodes_at_level) / 2)
    else:
        pos = nx.spring_layout(tree, k=2, iterations=50)
    
    # Draw edges
    nx.draw_networkx_edges(tree, pos, ax=ax, arrows=True, arrowsize=20,
                          edge_color='gray', width=2, alpha=0.6)
    
    # Draw nodes
    node_colors = ['red' if n == 0 else 'lightblue' for n in tree.nodes()]
    nx.draw_networkx_nodes(tree, pos, ax=ax, node_color=node_colors,
                          node_size=1000, alpha=0.8)
    
    # Draw labels
    labels = {n: f"C{n}\nT={tree.nodes[n].get('time', 0):.1f}" 
             for n in tree.nodes()}
    nx.draw_networkx_labels(tree, pos, labels, ax=ax, font_size=8)
    
    ax.set_title('Phylogenetic Tree', fontsize=14)
    ax.axis('off')
    
    return ax


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 20,
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plot feature importance.
    
    Args:
        importance_df: DataFrame with 'feature' and 'importance' columns
        top_n: Number of top features to show
        ax: Optional matplotlib axes
        
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    top_features = importance_df.head(top_n)
    
    ax.barh(range(len(top_features)), top_features['importance'].values)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'].values)
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title(f'Top {top_n} Feature Importances', fontsize=14)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    
    return ax


def plot_prediction_timeline(
    mutations_df: pd.DataFrame,
    predictions: pd.DataFrame,
    patient_id: str,
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plot predicted vs actual mutation emergence timeline.
    
    Args:
        mutations_df: Original mutations dataframe
        predictions: Predictions dataframe
        patient_id: Patient identifier
        ax: Optional matplotlib axes
        
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    patient_mutations = mutations_df[mutations_df['patient_id'] == patient_id]
    patient_preds = predictions[predictions['patient_id'] == patient_id]
    
    # Get actual emergence times
    actual_times = patient_mutations.groupby('mutation_id')['emergence_time_days'].first()
    
    # Get predicted probabilities (or times)
    if 'predicted_emerge_prob' in patient_preds.columns:
        predicted_scores = patient_preds.groupby('mutation_id')['predicted_emerge_prob'].mean()
        # Convert probability to "predicted time" (higher prob = earlier)
        predicted_times = -predicted_scores  # Invert for visualization
    else:
        predicted_times = patient_preds.groupby('mutation_id')['predicted_vaf_increase'].mean()
    
    # Plot
    mutations = list(set(actual_times.index) & set(predicted_times.index))
    
    if len(mutations) > 0:
        actual_values = [actual_times[m] for m in mutations]
        predicted_values = [predicted_times[m] for m in mutations]
        
        # Normalize for comparison
        if len(actual_values) > 1:
            actual_norm = (np.array(actual_values) - min(actual_values)) / (max(actual_values) - min(actual_values) + 1e-6)
            predicted_norm = (np.array(predicted_values) - min(predicted_values)) / (max(predicted_values) - min(predicted_values) + 1e-6)
        else:
            actual_norm = np.array([0.5])
            predicted_norm = np.array([0.5])
        
        x_pos = np.arange(len(mutations))
        width = 0.35
        
        ax.bar(x_pos - width/2, actual_norm, width, label='Actual', alpha=0.7)
        ax.bar(x_pos + width/2, predicted_norm, width, label='Predicted', alpha=0.7)
        
        ax.set_xlabel('Mutation', fontsize=12)
        ax.set_ylabel('Normalized Time', fontsize=12)
        ax.set_title(f'Predicted vs Actual Emergence Timeline (Patient: {patient_id})', fontsize=14)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([m[:10] for m in mutations], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    return ax


def create_summary_dashboard(
    mutations_df: pd.DataFrame,
    clone_data: pd.DataFrame,
    patient_id: str,
    predictions: Optional[pd.DataFrame] = None,
    save_path: Optional[str] = None
):
    """
    Create a comprehensive dashboard for a patient.
    
    Args:
        mutations_df: Mutations dataframe
        clone_data: Clone dataframe
        patient_id: Patient identifier
        predictions: Optional predictions dataframe
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Clone timeline
    plot_clone_timeline(clone_data, patient_id, ax=axes[0, 0])
    
    # VAF trajectories for top mutations
    patient_mutations = mutations_df[mutations_df['patient_id'] == patient_id]
    top_mutations = patient_mutations.groupby('mutation_id')['vaf'].max().nlargest(3).index
    
    for i, mut_id in enumerate(top_mutations[:3]):
        plot_vaf_trajectory(mutations_df, mut_id, patient_id, ax=axes[0, 1])
    
    # Predictions if available
    if predictions is not None:
        plot_prediction_timeline(mutations_df, predictions, patient_id, ax=axes[1, 0])
    
    # Summary statistics
    axes[1, 1].axis('off')
    summary_text = f"Patient: {patient_id}\n\n"
    summary_text += f"Total mutations: {patient_mutations['mutation_id'].nunique()}\n"
    summary_text += f"Total clones: {clone_data[clone_data['patient_id'] == patient_id]['clone_id'].nunique()}\n"
    summary_text += f"Timepoints: {patient_mutations['timepoint'].nunique()}\n"
    
    if predictions is not None:
        patient_preds = predictions[predictions['patient_id'] == patient_id]
        if 'predicted_emerge_prob' in patient_preds.columns:
            summary_text += f"\nPredicted emerging mutations: {len(patient_preds[patient_preds['predicted_emerge_prob'] > 0.5])}\n"
    
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
                    family='monospace')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


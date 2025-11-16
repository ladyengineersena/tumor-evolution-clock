"""
Synthetic clonal trajectory generator for tumor evolution simulation.

This script generates realistic synthetic data simulating:
- Phylogenetic trees of tumor clones
- VAF (Variant Allele Frequency) trajectories over time
- Multi-sample/multi-timepoint data
"""

import argparse
import json
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import networkx as nx
from datetime import datetime, timedelta


class ClonalTrajectorySimulator:
    """Simulates tumor clonal evolution trajectories."""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self.rng = np.random.default_rng(seed)
    
    def generate_phylogenetic_tree(self, n_clones: int) -> nx.DiGraph:
        """
        Generate a random phylogenetic tree for tumor clones.
        
        Args:
            n_clones: Number of clones to generate
            
        Returns:
            Directed graph representing the phylogenetic tree
        """
        tree = nx.DiGraph()
        tree.add_node(0, name="root", mutations=[], time=0.0)
        
        # Generate clones with parent-child relationships
        for i in range(1, n_clones):
            # Randomly select a parent from existing clones
            parent = self.rng.integers(0, i)
            time = tree.nodes[parent]['time'] + self.rng.exponential(0.3)
            
            # Assign mutations to this clone
            n_mutations = self.rng.poisson(5) + 1
            mutations = list(range(len(tree.nodes) * 10, len(tree.nodes) * 10 + n_mutations))
            
            tree.add_node(i, name=f"clone_{i}", mutations=mutations, time=time)
            tree.add_edge(parent, i)
        
        return tree
    
    def simulate_vaf_trajectory(
        self, 
        tree: nx.DiGraph,
        clone_id: int,
        timepoints: np.ndarray,
        growth_rate: float = None
    ) -> np.ndarray:
        """
        Simulate VAF trajectory for a clone over time.
        
        Args:
            tree: Phylogenetic tree
            clone_id: ID of the clone
            timepoints: Array of time points
            growth_rate: Growth rate (if None, random)
            
        Returns:
            VAF values at each timepoint
        """
        if growth_rate is None:
            growth_rate = self.rng.uniform(0.01, 0.1)
        
        clone_time = tree.nodes[clone_id]['time']
        vaf = np.zeros_like(timepoints, dtype=float)
        
        for i, t in enumerate(timepoints):
            if t < clone_time:
                vaf[i] = 0.0
            else:
                # Logistic growth model
                time_since_emergence = t - clone_time
                max_vaf = self.rng.uniform(0.1, 0.8)
                vaf[i] = max_vaf / (1 + np.exp(-growth_rate * time_since_emergence * 10))
                vaf[i] += self.rng.normal(0, 0.02)  # Add noise
                vaf[i] = np.clip(vaf[i], 0, 1)
        
        return vaf
    
    def generate_patient_data(
        self,
        patient_id: str,
        n_timepoints: int = 3,
        n_clones: int = 5,
        time_span_days: int = 365
    ) -> Dict:
        """
        Generate complete patient data with clonal trajectories.
        
        Args:
            patient_id: Unique patient identifier
            n_timepoints: Number of sampling timepoints
            n_clones: Number of clones
            time_span_days: Total time span in days
            
        Returns:
            Dictionary containing patient data
        """
        # Generate phylogenetic tree
        tree = self.generate_phylogenetic_tree(n_clones)
        
        # Generate timepoints
        baseline_date = datetime(2020, 1, 1)
        timepoints_days = np.linspace(0, time_span_days, n_timepoints)
        timepoints = np.array([baseline_date + timedelta(days=int(d)) for d in timepoints_days])
        
        # Collect all mutations
        all_mutations = []
        for node in tree.nodes():
            all_mutations.extend(tree.nodes[node]['mutations'])
        
        # Generate VAF data for each mutation at each timepoint
        mutation_data = []
        clone_data = []
        
        for clone_id in tree.nodes():
            clone_info = tree.nodes[clone_id]
            mutations = clone_info['mutations']
            emergence_time = clone_info['time']
            
            # Simulate clone VAF trajectory
            clone_vaf = self.simulate_vaf_trajectory(tree, clone_id, timepoints_days)
            
            clone_data.append({
                'clone_id': clone_id,
                'parent_id': list(tree.predecessors(clone_id))[0] if list(tree.predecessors(clone_id)) else None,
                'emergence_time_days': emergence_time,
                'mutations': mutations,
                'vaf_trajectory': clone_vaf.tolist()
            })
            
            # For each mutation in this clone
            for mut_id in mutations:
                # Mutation VAF is approximately clone VAF (with some noise)
                mut_vaf = clone_vaf.copy()
                mut_vaf += self.rng.normal(0, 0.01, size=len(mut_vaf))
                mut_vaf = np.clip(mut_vaf, 0, 1)
                
                mutation_data.append({
                    'mutation_id': f"mut_{mut_id}",
                    'clone_id': clone_id,
                    'gene': f"GENE_{mut_id % 50}",  # Simulate ~50 genes
                    'timepoint': list(range(n_timepoints)),
                    'vaf': mut_vaf.tolist(),
                    'ccf': mut_vaf.tolist(),  # Simplified: CCF ≈ VAF
                    'emergence_time_days': emergence_time
                })
        
        # Generate pathway information (simplified)
        pathways = ['PI3K', 'MAPK', 'p53', 'Cell_Cycle', 'DNA_Repair']
        pathway_activation = {}
        for pathway in pathways:
            # Randomly assign pathway activation to some clones
            active_clones = self.rng.choice(list(tree.nodes()), 
                                           size=self.rng.integers(1, n_clones//2 + 1),
                                           replace=False)
            pathway_activation[pathway] = [int(c) for c in active_clones]
        
        # Treatment simulation (optional)
        treatment_start_day = self.rng.integers(30, time_span_days // 2)
        treatment_type = self.rng.choice(['Chemotherapy', 'Targeted', 'Immunotherapy', 'None'])
        
        return {
            'patient_id': patient_id,
            'baseline_date': baseline_date.isoformat(),
            'timepoints_days': timepoints_days.tolist(),
            'timepoints': [t.isoformat() for t in timepoints],
            'n_clones': n_clones,
            'phylogenetic_tree': self._tree_to_dict(tree),
            'clones': clone_data,
            'mutations': mutation_data,
            'pathway_activation': pathway_activation,
            'treatment': {
                'start_day': int(treatment_start_day),
                'type': treatment_type
            },
            'metadata': {
                'cancer_type': self.rng.choice(['Lung', 'Breast', 'Colorectal', 'Melanoma']),
                'age': self.rng.integers(40, 80),
                'sex': self.rng.choice(['M', 'F'])
            }
        }
    
    def _tree_to_dict(self, tree: nx.DiGraph) -> Dict:
        """Convert NetworkX tree to serializable dictionary."""
        return {
            'nodes': {str(n): {'mutations': tree.nodes[n]['mutations'],
                              'time': float(tree.nodes[n]['time'])}
                     for n in tree.nodes()},
            'edges': [(int(u), int(v)) for u, v in tree.edges()]
        }
    
    def generate_cohort(
        self,
        n_patients: int = 50,
        output_dir: str = "data/synthetic",
        **kwargs
    ):
        """
        Generate a cohort of patients.
        
        Args:
            n_patients: Number of patients to generate
            output_dir: Output directory
            **kwargs: Additional arguments for generate_patient_data
        """
        os.makedirs(output_dir, exist_ok=True)
        
        all_patients = []
        for i in range(n_patients):
            patient_id = f"PATIENT_{i:04d}"
            patient_data = self.generate_patient_data(patient_id, **kwargs)
            all_patients.append(patient_data)
            
            # Save individual patient file
            patient_file = os.path.join(output_dir, f"{patient_id}.json")
            with open(patient_file, 'w', encoding='utf-8') as f:
                json.dump(patient_data, f, indent=2, ensure_ascii=False)
        
        # Save cohort summary
        cohort_file = os.path.join(output_dir, "cohort_summary.json")
        cohort_summary = {
            'n_patients': n_patients,
            'patient_ids': [p['patient_id'] for p in all_patients],
            'generation_date': datetime.now().isoformat()
        }
        with open(cohort_file, 'w', encoding='utf-8') as f:
            json.dump(cohort_summary, f, indent=2, ensure_ascii=False)
        
        # Save as CSV for easy loading
        self._save_as_dataframes(all_patients, output_dir)
        
        print(f"Generated {n_patients} patients in {output_dir}")
        return all_patients
    
    def _save_as_dataframes(self, patients: List[Dict], output_dir: str):
        """Save data as pandas DataFrames (CSV format)."""
        # Mutations dataframe
        mutation_rows = []
        for patient in patients:
            for mut in patient['mutations']:
                for tp_idx, vaf in enumerate(mut['vaf']):
                    mutation_rows.append({
                        'patient_id': patient['patient_id'],
                        'mutation_id': mut['mutation_id'],
                        'clone_id': mut['clone_id'],
                        'gene': mut['gene'],
                        'timepoint': tp_idx,
                        'timepoint_days': patient['timepoints_days'][tp_idx],
                        'vaf': vaf,
                        'ccf': mut['ccf'][tp_idx],
                        'emergence_time_days': mut['emergence_time_days']
                    })
        
        mutations_df = pd.DataFrame(mutation_rows)
        mutations_df.to_csv(os.path.join(output_dir, "mutations.csv"), index=False, encoding='utf-8')
        
        # Clones dataframe
        clone_rows = []
        for patient in patients:
            for clone in patient['clones']:
                for tp_idx, vaf in enumerate(clone['vaf_trajectory']):
                    clone_rows.append({
                        'patient_id': patient['patient_id'],
                        'clone_id': clone['clone_id'],
                        'parent_id': clone['parent_id'],
                        'timepoint': tp_idx,
                        'timepoint_days': patient['timepoints_days'][tp_idx],
                        'vaf': vaf,
                        'emergence_time_days': clone['emergence_time_days']
                    })
        
        clones_df = pd.DataFrame(clone_rows)
        clones_df.to_csv(os.path.join(output_dir, "clones.csv"), index=False, encoding='utf-8')
        
        # Patients metadata
        patient_rows = []
        for patient in patients:
            patient_rows.append({
                'patient_id': patient['patient_id'],
                'cancer_type': patient['metadata']['cancer_type'],
                'age': patient['metadata']['age'],
                'sex': patient['metadata']['sex'],
                'treatment_type': patient['treatment']['type'],
                'treatment_start_day': patient['treatment']['start_day'],
                'n_clones': patient['n_clones']
            })
        
        patients_df = pd.DataFrame(patient_rows)
        patients_df.to_csv(os.path.join(output_dir, "patients.csv"), index=False, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic clonal trajectory data')
    parser.add_argument('--out', type=str, default='data/synthetic',
                       help='Output directory')
    parser.add_argument('--n_patients', type=int, default=50,
                       help='Number of patients to generate')
    parser.add_argument('--n_timepoints', type=int, default=3,
                       help='Number of timepoints per patient')
    parser.add_argument('--n_clones', type=int, default=5,
                       help='Number of clones per patient')
    parser.add_argument('--time_span_days', type=int, default=365,
                       help='Time span in days')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    simulator = ClonalTrajectorySimulator(seed=args.seed)
    simulator.generate_cohort(
        n_patients=args.n_patients,
        output_dir=args.out,
        n_timepoints=args.n_timepoints,
        n_clones=args.n_clones,
        time_span_days=args.time_span_days
    )


if __name__ == '__main__':
    main()


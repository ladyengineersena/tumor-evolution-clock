"""
Quick test script to verify the pipeline works.
Run this to test the complete pipeline.
"""

import os
import sys
import subprocess

def test_pipeline():
    """Test the complete pipeline."""
    print("=" * 60)
    print("Testing Tumor Evolution Clock Pipeline")
    print("=" * 60)
    
    # Step 1: Generate synthetic data
    print("\n[1/4] Generating synthetic data...")
    try:
        from scripts.simulate_clonal_trajectories import ClonalTrajectorySimulator
        os.makedirs('data/synthetic', exist_ok=True)
        simulator = ClonalTrajectorySimulator(seed=42)
        patients = simulator.generate_cohort(
            n_patients=10,
            output_dir='data/synthetic',
            n_timepoints=3,
            n_clones=5,
            time_span_days=365
        )
        print(f"✓ Generated {len(patients)} patients")
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
    
    # Step 2: Test feature extraction
    print("\n[2/4] Testing feature extraction...")
    try:
        import pandas as pd
        from src.features import FeatureExtractor, create_target_labels
        
        mutations_df = pd.read_csv('data/synthetic/mutations.csv')
        clones_df = pd.read_csv('data/synthetic/clones.csv')
        patients_df = pd.read_csv('data/synthetic/patients.csv')
        
        targets_df = create_target_labels(mutations_df, prediction_horizon=1)
        extractor = FeatureExtractor()
        
        # Test on one patient
        test_patient = patients_df.iloc[0]['patient_id']
        features = extractor.extract_patient_features(
            mutations_df, clones_df, patients_df,
            patient_id=test_patient, timepoint=0
        )
        print(f"✓ Extracted {len(features)} feature rows for test patient")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 3: Test model training
    print("\n[3/4] Testing model training...")
    try:
        from src.models import TumorEvolutionPredictor
        from sklearn.model_selection import train_test_split
        
        # Prepare minimal training data
        features_list = []
        for idx, target_row in targets_df.head(50).iterrows():
            patient_id = target_row['patient_id']
            timepoint = target_row['timepoint']
            
            patient_features = extractor.extract_patient_features(
                mutations_df, clones_df, patients_df,
                patient_id=patient_id, timepoint=timepoint
            )
            
            mutation_features = patient_features[
                patient_features['mutation_id'] == target_row['mutation_id']
            ]
            
            if len(mutation_features) > 0:
                feature_row = mutation_features.iloc[0].to_dict()
                feature_row['target_emerge'] = target_row['target_emerge']
                features_list.append(feature_row)
        
        if len(features_list) == 0:
            print("⚠ No features extracted, skipping model test")
        else:
            features_df = pd.DataFrame(features_list)
            feature_cols = [col for col in features_df.columns 
                          if col not in ['patient_id', 'mutation_id', 'timepoint', 'target_emerge']]
            
            X = features_df[feature_cols].fillna(0).replace([float('inf'), float('-inf')], 0)
            y = features_df['target_emerge'].values
            
            if len(X) > 5:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=42
                )
                
                model = TumorEvolutionPredictor(model_type='xgboost', task='classification')
                model.fit(X_train, y_train, feature_names=feature_cols)
                
                score = model.model.score(model.scaler.transform(X_test), y_test)
                print(f"✓ Model trained, test accuracy: {score:.4f}")
            else:
                print("⚠ Insufficient data for model training")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 4: Test visualization
    print("\n[4/4] Testing visualization...")
    try:
        from src.visualization import plot_clone_timeline
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        
        test_patient = patients_df.iloc[0]['patient_id']
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_clone_timeline(clones_df, test_patient, ax=ax)
        plt.close(fig)
        print("✓ Visualization functions work")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
    return True


if __name__ == '__main__':
    success = test_pipeline()
    sys.exit(0 if success else 1)


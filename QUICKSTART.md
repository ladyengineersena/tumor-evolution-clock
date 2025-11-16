# Quick Start Guide

## Installation

1. **Create virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Quick Test

Run the test script to verify everything works:

```bash
python test_pipeline.py
```

This will:
- Generate synthetic data
- Test feature extraction
- Train a simple model
- Test visualization

## Full Pipeline

### 1. Generate Synthetic Data

```bash
python scripts/simulate_clonal_trajectories.py --out data/synthetic --n_patients 50
```

Options:
- `--n_patients`: Number of patients (default: 50)
- `--n_timepoints`: Number of timepoints per patient (default: 3)
- `--n_clones`: Number of clones per patient (default: 5)
- `--time_span_days`: Time span in days (default: 365)
- `--seed`: Random seed (default: 42)

### 2. Train Model

```bash
python src/train.py --data data/synthetic --out outputs/model.pkl
```

Options:
- `--data`: Directory containing data files
- `--out`: Output path for model
- `--model_type`: Model type (`xgboost` or `random_forest`, default: `xgboost`)
- `--task`: Task type (`classification` or `regression`, default: `classification`)
- `--target`: Target variable (`target_emerge` or `target_vaf_increase`, default: `target_emerge`)

### 3. Evaluate Model

```bash
python src/evaluate.py --model outputs/model.pkl --data data/synthetic
```

Options:
- `--model`: Path to trained model
- `--data`: Directory containing data files
- `--out`: Output path for results (default: `outputs/evaluation_results.json`)
- `--target`: Target variable

## Using Jupyter Notebooks

1. **Start Jupyter:**
   ```bash
   jupyter notebook
   ```

2. **Open notebooks in order:**
   - `notebooks/01_simulate_data.ipynb` - Data generation
   - `notebooks/02_clone_deconv.ipynb` - Clone deconvolution
   - `notebooks/03_modeling.ipynb` - Model training and evaluation

## Output Files

After running the pipeline, you'll have:

- `data/synthetic/`:
  - `mutations.csv` - Mutation data
  - `clones.csv` - Clone data
  - `patients.csv` - Patient metadata
  - `cohort_summary.json` - Cohort summary
  - Individual patient JSON files

- `outputs/`:
  - `model.pkl` - Trained model
  - `model_metrics.json` - Training metrics
  - `model_feature_importance.csv` - Feature importance
  - `evaluation_results.json` - Evaluation metrics

## Troubleshooting

### Import Errors

If you get import errors, make sure you're in the project root directory and the virtual environment is activated.

### Memory Issues

For large datasets, reduce `--n_patients` or process in batches.

### Missing Dependencies

Make sure all packages in `requirements.txt` are installed:
```bash
pip install -r requirements.txt
```

## Next Steps

- Explore the notebooks for detailed examples
- Modify feature extraction in `src/features.py`
- Experiment with different models in `src/models.py`
- Add custom visualizations in `src/visualization.py`


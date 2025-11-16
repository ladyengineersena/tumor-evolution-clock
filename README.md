# Tumor Evolution Clock — Predicting Tumor Molecular Timelines from Multi-sample Genomics

## Overview

This project develops an AI system to predict **evolutionary timelines** of tumors from multi-sample biopsy/genome/mutation frequency data. The system estimates:
- Which mutations/subclones emerged first
- Which pathways/mechanisms will activate in sequence
- Probable time intervals for these events

## Clinical Value

While existing studies typically answer "which mutations are present?", this system addresses the critical questions of **timing** and **future pathway activation**, providing valuable insights for:
- Treatment planning (which targets to prioritize and when)
- Resistance prediction
- Combination therapy selection

## ⚠️ Important Notice

**This is a research prototype only. NOT for clinical decision-making.**

See [ETHICS.md](ETHICS.md) for detailed ethical and legal considerations.

## Repository Structure

```
tumor-evolution-clock/
├── data/
│   └── synthetic/          # Generated synthetic data
├── notebooks/
│   ├── 01_simulate_data.ipynb
│   ├── 02_clone_deconv.ipynb
│   └── 03_modeling.ipynb
├── scripts/
│   └── simulate_clonal_trajectories.py
├── src/
│   ├── features.py          # Feature engineering
│   ├── models.py            # ML models
│   ├── train.py             # Training pipeline
│   ├── evaluate.py          # Evaluation metrics
│   └── visualization.py    # Plotting tools
├── outputs/                 # Model outputs
├── README.md
├── ETHICS.md
└── requirements.txt
```

## Quick Start

### 1. Setup Environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Generate Synthetic Data

```bash
python scripts/simulate_clonal_trajectories.py --out data/synthetic --n_patients 50
```

### 3. Train Baseline Model

```bash
python src/train.py --data data/synthetic --out outputs/model.pkl
```

### 4. Evaluate Model

```bash
python src/evaluate.py --model outputs/model.pkl --data data/synthetic
```

### 5. Explore Notebooks

Open the Jupyter notebooks in `notebooks/` for interactive exploration:
- `01_simulate_data.ipynb` - Data generation demonstration
- `02_clone_deconv.ipynb` - Clonal decomposition demo
- `03_modeling.ipynb` - Model training and prediction examples

## Methodology

The system combines multiple approaches:

### A) Phylogenetic + Bayesian Approach
- Clone deconvolution from VAF data
- Phylogenetic tree inference
- Bayesian timing estimation

### B) Machine Learning / Deep Learning
- Temporal sequence models (LSTM/Transformer)
- Graph-based models (GNN)
- Hybrid approaches combining phylogenetic and ML methods

### C) Feature Engineering
- VAF trajectories and CCF trends
- Pathway-level activation scores
- Clonal growth rate estimates
- Treatment features

## Performance Metrics
- **Order Accuracy (Kendall tau)**: Predicted vs actual mutation order
- **Time MAE (days)**: Average absolute error of predicted emergence times
- **Top-1/Top-3 Accuracy**: Pathway activation prediction
- **C-index**: Time-to-event concordance

## Data Requirements

### Input Data Types
- Multi-timepoint/multi-region biopsy data
- Somatic mutation lists with VAF (variant allele frequency)
- Copy-number profiles
- Structural variants (optional)
- RNA expression (optional)
- Methylation data (optional)

### Clinical Metadata
- Treatment type and start time
- Response information (responder/non-responder)
- Time-to-progression

## Ethical Considerations

- **IRB/Ethics approval required** for real patient data
- **No real patient data** in this repository (synthetic data only)
- Personal identifiers must be removed
- Dates converted to relative time (days since baseline)
- Fairness reporting required (performance across subgroups)

See [ETHICS.md](ETHICS.md) for complete details.

## License

All Rights Reserved (see LICENSE file or repository settings).

## Contributing

This is a research prototype. For questions or collaboration, please contact the maintainers.

## Citation

If you use this code in your research, please cite appropriately.

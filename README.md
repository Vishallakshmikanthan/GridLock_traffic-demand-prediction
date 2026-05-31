# Traffic Demand Prediction - Hackathon Project

## Overview
This repository contains the codebase and workspace for the Traffic Demand Prediction machine learning hackathon. 

## Best Practices Structure
- `data/`: Contains all datasets.
  - `raw/`: Read-only, original data dump (NEVER edit these files).
  - `processed/`: Cleaned and transformed data ready for modeling.
- `notebooks/`: Jupyter notebooks for Exploratory Data Analysis (EDA) and experimental modeling. Number them sequentially (e.g., `01_EDA.ipynb`).
- `src/`: Reusable Python source code for data processing, feature engineering, and model training. Kept separate from notebooks to maintain clean code practices.
- `submissions/`: Output prediction files (usually `.csv`) formatted and ready for competition submission.

## Getting Started

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Load Data**:
   Place your raw training and testing data files inside the `data/raw/` folder.

3. **Explore & Model**:
   - Begin your initial analysis in `notebooks/01_EDA.ipynb`.
   - Build out reusable logic in `src/features.py`.
   - Run your model training pipeline using `src/train.py`.

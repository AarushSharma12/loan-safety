# 🎯 Loan Safety ML Platform

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4.2-orange)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Predict loan risk with state-of-the-art machine learning models. Compare **Decision Trees** and **Random Forests** to identify safe vs risky loans using LendingClub data.

## ✨ Features

- **🌲 Decision Trees & 🌳 Random Forests** - Choose the right model for your needs
- **📊 Comprehensive Evaluation** - ROC curves, confusion matrices, feature importance
- **🔮 Real-time Predictions** - Interactive web interface for instant risk assessment
- **🏆 Model Comparison** - Side-by-side performance metrics to pick the best model
- **📈 Business Metrics** - Understand financial impact of predictions
- **🎯 High Interpretability** - Understand exactly why loans are classified as risky

## 🚀 Quick Start

```bash
# 1. Setup environment
python -m venv .venv && source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Train your first model (Random Forest)
python src/train.py --config configs/config.yaml --model random_forest

# 3. Launch the web interface
streamlit run app.py
```

## 🛠️ Installation

### Option 1: Make Commands (Recommended)

```bash
# Complete setup
make setup

# Train models
make train           # Uses config default (Random Forest)
make train-dt        # Train Decision Tree
make train-rf        # Train Random Forest

# Evaluate
make eval-latest     # Evaluate most recent model
make eval MODEL=artifacts/model_*.joblib

# Launch web app
make app
```

### Option 2: Python Commands

```bash
# Setup
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Train
python src/train.py --config configs/config.yaml --model random_forest
python src/train.py --config configs/config.yaml --model decision_tree

# Evaluate
python src/evaluate.py --model artifacts/model_*.joblib --config configs/config.yaml

# Web interface
streamlit run app.py
```

## 🏗️ Project Structure

```
loan-safety-ml/
├── configs/
│   └── config.yaml          # Model & training configuration
├── src/
│   ├── train.py            # Training pipeline for DT & RF
│   ├── evaluate.py         # Comprehensive evaluation metrics
│   ├── predict.py          # Inference script
│   ├── data.py             # Data loading & preprocessing
│   └── features.py         # Feature engineering pipeline
├── app.py                  # Streamlit web interface
├── artifacts/              # Saved models (created automatically)
├── outputs/                # Evaluation results & plots
├── data/
│   └── raw/
│       └── lending-club-data.csv  # Your dataset here
└── tests/                  # Unit tests
```

## 📊 Model Comparison

| Model             | Pros                                                                 | Cons                                                       | Best For                             |
| ----------------- | -------------------------------------------------------------------- | ---------------------------------------------------------- | ------------------------------------ |
| **Decision Tree** | • Fast training<br>• Highly interpretable<br>• Low memory usage      | • Prone to overfitting<br>• Lower accuracy                 | Quick prototypes, Explainable AI     |
| **Random Forest** | • Higher accuracy<br>• Robust to overfitting<br>• Feature importance | • Slower training<br>• More memory<br>• Less interpretable | Production systems, Best performance |

### Performance Benchmarks\*

| Metric    | Decision Tree | Random Forest | Improvement |
| --------- | ------------- | ------------- | ----------- |
| Accuracy  | 0.82          | 0.89          | +8.5%       |
| Precision | 0.80          | 0.87          | +8.8%       |
| Recall    | 0.85          | 0.91          | +7.1%       |
| F1 Score  | 0.82          | 0.89          | +8.5%       |
| ROC AUC   | 0.81          | 0.94          | +16.0%      |

\*Results may vary based on dataset and hyperparameters

## 🎯 Features Used

### Categorical Features (8)

- `grade` - Loan grade assigned by LendingClub
- `sub_grade` - Loan sub-grade for finer risk assessment
- `home_ownership` - Rent, own, mortgage, etc.
- `purpose` - Reason for loan (debt consolidation, etc.)
- `term` - Loan duration (36 or 60 months)
- `short_emp` - Short employment flag
- `last_delinq_none` - No recent delinquencies
- `last_major_derog_none` - No major derogatory marks

### Numerical Features (4)

- `emp_length_num` - Employment length in years
- `dti` - Debt-to-income ratio
- `revol_util` - Revolving line utilization rate
- `total_rec_late_fee` - Total late fees received

## 📈 Evaluation Metrics

The platform provides comprehensive evaluation:

- **Classification Metrics**: Accuracy, Precision, Recall, F1 Score
- **Probabilistic Metrics**: ROC AUC, Log Loss
- **Business Metrics**:
  - False Positives (risky loans approved) → Potential losses
  - False Negatives (safe loans rejected) → Missed opportunities
- **Visualizations**:
  - Confusion Matrix
  - ROC Curve
  - Feature Importance Plot
  - Cross-validation scores

## 🔮 Making Predictions

### Via Web Interface

1. Launch app: `streamlit run app.py`
2. Select model and enter loan features
3. Get instant risk assessment

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- LendingClub for providing the loan dataset
- scikit-learn community for excellent ML tools
- Streamlit for the amazing web framework

## 📞 Contact

**Author**: Aarush Sharma

---

<p align="center">Made with Python</p>

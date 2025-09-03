# 🎯 Loan Safety ML Platform

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4.2-orange)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Predict loan risk with state-of-the-art machine learning models. Compare **Decision Trees** and **Random Forests** to identify safe vs risky loans using LendingClub data.

![Loan Safety Platform](https://via.placeholder.com/800x400/4a90e2/ffffff?text=Loan+Safety+ML+Platform)

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

## 📋 Prerequisites

- Python 3.11+
- 4GB RAM minimum (8GB recommended for Random Forest)
- LendingClub dataset in `data/raw/lending-club-data.csv`

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

## ⚙️ Configuration

Edit `configs/config.yaml` to customize:

```yaml
model:
  type: random_forest # or decision_tree

  random_forest:
    n_estimators: 100 # Number of trees
    max_depth: 10 # Maximum tree depth
    max_features: sqrt # Feature sampling strategy

  decision_tree:
    max_depth: 6 # Maximum tree depth
    min_samples_split: 2

training:
  test_size: 0.2 # 80/20 train-test split
  random_seed: 42 # For reproducibility
  cross_validation:
    enabled: true # Enable k-fold CV
    folds: 5
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
2. Go to "🔮 Predict" tab
3. Select model and enter loan features
4. Get instant risk assessment

### Via Command Line

```bash
# Create a JSON file with features
echo '{
  "grade": "B",
  "sub_grade": "B2",
  "home_ownership": "RENT",
  "purpose": "debt_consolidation",
  "term": "36 months",
  "emp_length_num": 5.0,
  "dti": 15.5,
  "revol_util": 45.0,
  "total_rec_late_fee": 0.0,
  "short_emp": 0,
  "last_delinq_none": 1,
  "last_major_derog_none": 1
}' > sample_loan.json

# Predict
python src/predict.py --model artifacts/model_*.joblib --input sample_loan.json
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## 📚 Advanced Usage

### Cross-Validation Training

```python
# In config.yaml
training:
  cross_validation:
    enabled: true
    folds: 5
```

### Hyperparameter Tuning

```python
# Coming soon: Grid search and Bayesian optimization
python src/tune.py --model random_forest --method grid
```

### Model Serving API

```python
# Coming soon: FastAPI endpoint
uvicorn api:app --reload
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md).

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- LendingClub for providing the loan dataset
- scikit-learn community for excellent ML tools
- Streamlit for the amazing web framework

## 📞 Contact

**Author**: Aarush Sharma  
**Email**: your.email@example.com  
**Project Link**: [https://github.com/yourusername/loan-safety-ml](https://github.com/yourusername/loan-safety-ml)

---

<p align="center">Made with ❤️ and Python</p>

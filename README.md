# Fraud Detection System

Real-time fraud detection pipeline built with XGBoost and FastAPI, 
trained on 284,000 real credit card transactions.

## Live Dashboard
[paste your screenshot here]

## Model Performance
| Metric    | Score |
|-----------|-------|
| Recall    | 87%   |
| Precision | 26%   |
| F1 Score  | 0.39  |
| Latency   | ~9ms  |

## Architecture
creditcard.csv → train.py → fraud_model.pkl
↓
Transaction → api.py (FastAPI) → risk score + decision
↓
dashboard.py (Streamlit)

## How to Run

**1. Train the model**
```bash
python train.py
```

**2. Start the API**
```bash
uvicorn api:app --reload
```

**3. Start the dashboard**
```bash
streamlit run dashboard.py
```

## Tech Stack
- Python, XGBoost, scikit-learn
- imbalanced-learn (SMOTE)
- FastAPI, Uvicorn
- Streamlit
- pandas, numpy
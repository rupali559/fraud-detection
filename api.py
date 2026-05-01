import pickle
import time
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Fraud Detection API")

print("Loading model...")
model = pickle.load(open('models/fraud_model.pkl', 'rb'))
scaler = pickle.load(open('models/scaler.pkl', 'rb'))
print("Model loaded.")

class Transaction(BaseModel):
    Time: float
    V1: float; V2: float; V3: float; V4: float; V5: float
    V6: float; V7: float; V8: float; V9: float; V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float
    V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float
    V26: float; V27: float; V28: float
    Amount: float

@app.get("/")
def root():
    return {"status": "Fraud Detection API is running"}

@app.post("/score")
def score_transaction(txn: Transaction):
    start = time.time()

    hour = (txn.Time % 86400) // 3600
    amount_log = np.log1p(txn.Amount)

    if txn.Amount <= 10:
        amount_bucket = 0
    elif txn.Amount <= 100:
        amount_bucket = 1
    elif txn.Amount <= 1000:
        amount_bucket = 2
    else:
        amount_bucket = 3

    amount_scaled = scaler.transform([[txn.Amount]])[0][0]
    time_scaled = scaler.transform([[txn.Time]])[0][0]

    features = np.array([[
        txn.V1, txn.V2, txn.V3, txn.V4, txn.V5,
        txn.V6, txn.V7, txn.V8, txn.V9, txn.V10,
        txn.V11, txn.V12, txn.V13, txn.V14, txn.V15,
        txn.V16, txn.V17, txn.V18, txn.V19, txn.V20,
        txn.V21, txn.V22, txn.V23, txn.V24, txn.V25,
        txn.V26, txn.V27, txn.V28,
        hour, amount_log, amount_bucket,
        amount_scaled, time_scaled
    ]])

    risk_score = float(model.predict_proba(features)[0][1]) * 100
    latency_ms = round((time.time() - start) * 1000, 1)

    if risk_score >= 70:
        decision = "block"
    elif risk_score >= 40:
        decision = "review"
    else:
        decision = "approve"

    return {
        "risk_score": round(risk_score, 1),
        "decision": decision,
        "latency_ms": latency_ms
    }
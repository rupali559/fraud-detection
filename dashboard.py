import streamlit as st
import requests
import random
import time
import pandas as pd

st.set_page_config(page_title="Fraud Detection Monitor", layout="wide")
st.title("Fraud Detection — Live Monitor")

if 'log' not in st.session_state:
    st.session_state.log = []

def generate_transaction():
    is_fraud = random.random() < 0.05
    return {
        "Time": random.uniform(0, 172792),
        "V1": random.gauss(-3.0 if is_fraud else 0, 1),
        "V2": random.gauss(3.0 if is_fraud else 0, 1),
        "V3": random.gauss(-3.0 if is_fraud else 0, 1),
        "V4": random.gauss(2.0 if is_fraud else 0, 1),
        "V5": random.gauss(0, 1),
        "V6": random.gauss(0, 1),
        "V7": random.gauss(-3.0 if is_fraud else 0, 1),
        "V8": random.gauss(0, 1),
        "V9": random.gauss(0, 1),
        "V10": random.gauss(-3.0 if is_fraud else 0, 1),
        "V11": random.gauss(0, 1),
        "V12": random.gauss(-3.0 if is_fraud else 0, 1),
        "V13": random.gauss(0, 1),
        "V14": random.gauss(-3.0 if is_fraud else 0, 1),
        "V15": random.gauss(0, 1),
        "V16": random.gauss(0, 1),
        "V17": random.gauss(-3.0 if is_fraud else 0, 1),
        "V18": random.gauss(0, 1),
        "V19": random.gauss(0, 1),
        "V20": random.gauss(0, 1),
        "V21": random.gauss(0, 1),
        "V22": random.gauss(0, 1),
        "V23": random.gauss(0, 1),
        "V24": random.gauss(0, 1),
        "V25": random.gauss(0, 1),
        "V26": random.gauss(0, 1),
        "V27": random.gauss(0, 1),
        "V28": random.gauss(0, 1),
        "Amount": random.uniform(5000, 9000) if is_fraud else random.uniform(1, 500)
    }

def score(txn):
    try:
        res = requests.post(
            "http://127.0.0.1:8000/score",
            json=txn,
            timeout=2
        )
        return res.json()
    except:
        return None

placeholder = st.empty()

while True:
    txn = generate_transaction()
    result = score(txn)

    if result:
        st.session_state.log.append({
            "Amount": f"${round(txn['Amount'], 2)}",
            "Risk Score": result['risk_score'],
            "Decision": result['decision'].upper(),
            "Latency": f"{result['latency_ms']}ms"
        })

    log = st.session_state.log
    total = len(log)
    blocked = sum(1 for r in log if r['Decision'] == 'BLOCK')
    reviews = sum(1 for r in log if r['Decision'] == 'REVIEW')

    with placeholder.container():
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Transactions", total)
        col2.metric("Blocked", blocked)
        col3.metric("Under Review", reviews)
        col4.metric("Block Rate", f"{round(blocked/max(total,1)*100, 1)}%")

        st.divider()

        df = pd.DataFrame(log[-25:]).iloc[::-1]

        def color_rows(row):
            if row['Decision'] == 'BLOCK':
                return ['background-color: #fee2e2; color: #991b1b'] * len(row)
            elif row['Decision'] == 'REVIEW':
                return ['background-color: #fef9c3; color: #854d0e'] * len(row)
            return [''] * len(row)

        st.dataframe(
        df.style.apply(color_rows, axis=1),
        width='stretch',
        hide_index=True
    )

    time.sleep(1)
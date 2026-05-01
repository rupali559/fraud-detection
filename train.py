import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import xgboost as xgb

print("Step 1: Loading data...")
df = pd.read_csv('data/creditcard.csv')
print(f"Shape: {df.shape}")
print(f"Fraud cases: {df['Class'].sum()} out of {len(df)}")

print("\nStep 2: Feature engineering...")
df['hour'] = (df['Time'] % 86400) // 3600
df['amount_log'] = np.log1p(df['Amount'])
df['amount_bucket'] = pd.cut(
    df['Amount'],
    bins=[-1, 10, 100, 1000, 999999],
    labels=[0, 1, 2, 3]
).astype(int)

print("\nStep 3: Scaling Amount and Time...")
scaler = StandardScaler()
df['amount_scaled'] = scaler.fit_transform(df[['Amount']])
df['time_scaled'] = scaler.fit_transform(df[['Time']])
df.drop(['Time', 'Amount'], axis=1, inplace=True)

print("\nStep 4: Splitting data...")
X = df.drop('Class', axis=1)
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

print("\nStep 5: Applying SMOTE...")
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
print(f"After SMOTE - Class counts: {pd.Series(y_train_res).value_counts().to_dict()}")

print("\nStep 6: Training XGBoost model...")
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    eval_metric='logloss'
)
model.fit(X_train_res, y_train_res)
print("Training complete.")

print("\nStep 7: Evaluating model...")
y_proba = model.predict_proba(X_test)[:, 1]
threshold = 0.3
y_pred = (y_proba >= threshold).astype(int)

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=['Legit', 'Fraud']))

print("--- Confusion Matrix ---")
cm = confusion_matrix(y_test, y_pred)
print(f"True Negatives:  {cm[0][0]}")
print(f"False Positives: {cm[0][1]}")
print(f"False Negatives: {cm[1][0]}")
print(f"True Positives:  {cm[1][1]}")

print("\nStep 8: Saving model and scaler...")
pickle.dump(model, open('models/fraud_model.pkl', 'wb'))
pickle.dump(scaler, open('models/scaler.pkl', 'wb'))
pickle.dump(list(X.columns), open('models/feature_names.pkl', 'wb'))
print("Saved: models/fraud_model.pkl")
print("Saved: models/scaler.pkl")
print("Saved: models/feature_names.pkl")

print("\nDone! Note your Fraud recall % for your resume.")
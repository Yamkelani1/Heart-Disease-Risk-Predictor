import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ---------------------------------------------------------
# 1. GENERATE & SAVE DATASET (heart.csv)
# ---------------------------------------------------------
np.random.seed(42)
n_samples = 300

data = {
    "age": np.random.randint(29, 78, size=n_samples),
    "sex": np.random.choice([0, 1], size=n_samples, p=[0.32, 0.68]),
    "cp": np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.47, 0.17, 0.28, 0.08]),
    "trestbps": np.random.randint(94, 200, size=n_samples),
    "chol": np.random.randint(126, 564, size=n_samples),
    "fbs": np.random.choice([0, 1], size=n_samples, p=[0.85, 0.15]),
    "restecg": np.random.choice([0, 1, 2], size=n_samples, p=[0.48, 0.48, 0.04]),
    "thalach": np.random.randint(71, 202, size=n_samples),
    "exang": np.random.choice([0, 1], size=n_samples, p=[0.67, 0.33]),
    "oldpeak": np.round(np.random.uniform(0.0, 6.2, size=n_samples), 1),
    "slope": np.random.choice([0, 1, 2], size=n_samples, p=[0.07, 0.46, 0.47]),
    "ca": np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.58, 0.22, 0.12, 0.08]),
    "thal": np.random.choice([1, 2, 3], size=n_samples, p=[0.06, 0.54, 0.40]),
}

df = pd.DataFrame(data)

# Formulate realistic risk score target
risk_score = (
        (df["age"] > 55).astype(int) * 1.2 +
        (df["cp"] > 0).astype(int) * 2.0 +
        (df["thalach"] < 140).astype(int) * 1.5 +
        (df["exang"] == 1).astype(int) * 1.8 +
        (df["oldpeak"] > 1.5).astype(int) * 1.5 +
        np.random.normal(0, 1, size=n_samples)
)
df["target"] = (risk_score > risk_score.median()).astype(int)

# Save heart.csv to root directory
df.to_csv("heart.csv", index=False)
print("✅ Successfully generated 'heart.csv' in project folder!\n")

# ---------------------------------------------------------
# 2. STEP 1: TRAIN BASELINE RANDOM FOREST MODEL
# ---------------------------------------------------------
X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Measure baseline accuracy
y_pred = rf_model.predict(X_test)
baseline_acc = accuracy_score(y_test, y_pred)

print("--- Step 1 Checkpoint ---")
print(f"Dataset Shape: {df.shape}")
print(f"Baseline Random Forest Accuracy: {baseline_acc * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Test single sample prediction
sample = X_test.iloc[[0]]
pred_class = rf_model.predict(sample)[0]
pred_proba = rf_model.predict_proba(sample)[0]

print("\n--- Raw Sample Prediction Output ---")
print(f"Predicted Risk Class (0=Low, 1=High): {pred_class}")
print(f"Raw Probabilities [Low, High]: {pred_proba}")

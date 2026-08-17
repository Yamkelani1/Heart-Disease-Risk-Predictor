import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    layout="centered"
)

# ---------------------------------------------------------
# Load Data & Cache Model Training
# ---------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("heart.csv")

@st.cache_resource
def train_models(_df):
    X = _df.drop(columns=["target"])
    y = _df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42)
    }

    results = {}
    for name, m in models.items():
        m.fit(X_train, y_train)
        y_pred = m.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = {"model": m, "accuracy": acc}

    return results, X.columns.tolist()


# Load and train
df = load_data()
all_results, feature_names = train_models(df)

# Find best model
best_name = max(all_results, key=lambda k: all_results[k]["accuracy"])

# Sidebar - model comparison
st.sidebar.header("Model Comparison")
comparison_data = {
    "Model": list(all_results.keys()),
    "Accuracy": [f"{all_results[k]['accuracy']:.1%}" for k in all_results],
}
comparison_df = pd.DataFrame(comparison_data)
st.sidebar.dataframe(comparison_df, hide_index=True)
st.sidebar.write(f"Best performer: {best_name}")

# Model selector
selected_model_name = st.sidebar.selectbox(
    "Choose prediction model",
    options=list(all_results.keys()),
    index=list(all_results.keys()).index(best_name)
)
model = all_results[selected_model_name]["model"]
accuracy = all_results[selected_model_name]["accuracy"]

st.sidebar.metric("Selected Model Accuracy", f"{accuracy:.1%}")
st.sidebar.write(f"Trained on {len(df)} patient records")

# ---------------------------------------------------------
# UI Header
# ---------------------------------------------------------
st.title("❤️ Heart Disease Risk Predictor")
st.write("Enter patient information below to predict heart disease risk.")
st.markdown("---")

# ---------------------------------------------------------
# Patient Information Form (Two-Column Layout)
# ---------------------------------------------------------
st.header("Patient Information")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=50)
    sex = st.selectbox(
        "Sex",
        options=[0, 1],
        format_func=lambda x: "Female" if x == 0 else "Male"
    )
    cp = st.selectbox(
        "Chest Pain Type",
        options=[0, 1, 2, 3],
        format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"][x]
    )
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=250, value=120)
    chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
    fbs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dl",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )
    restecg = st.selectbox(
        "Resting ECG Results",
        options=[0, 1, 2],
        format_func=lambda x: ["Normal", "ST-T Abnormality", "Left Ventricular Hypertrophy"][x]
    )

with col2:
    thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
    exang = st.selectbox(
        "Exercise Induced Angina",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )
    oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=6.2, value=1.0, step=0.1)
    slope = st.selectbox(
        "ST Slope",
        options=[0, 1, 2],
        format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x]
    )
    ca = st.selectbox("Major Vessels (0-3)", options=[0, 1, 2, 3])
    thal = st.selectbox(
        "Thalassemia",
        options=[1, 2, 3],
        format_func=lambda x: {1: "Normal", 2: "Fixed Defect", 3: "Reversable Defect"}.get(x, x)
    )

# Prepare DataFrame for Prediction
input_data = pd.DataFrame([[
    age, sex, cp, trestbps, chol, fbs, restecg,
    thalach, exang, oldpeak, slope, ca, thal
]], columns=feature_names)

# ---------------------------------------------------------
# Prediction Output Block
# ---------------------------------------------------------
if st.button("Predict Risk", type="primary"):
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    risk_percentage = probabilities[1] * 100

    st.divider()
    st.header("Prediction Results")
    st.caption(f"Using: {selected_model_name}")

    if prediction == 1:
        st.error(f"⚠️ **High Risk of Heart Disease** ({risk_percentage:.1f}% probability)")
    else:
        st.success(f"✅ **Low Risk of Heart Disease** ({(100 - risk_percentage):.1f}% confidence)")

    # Risk level indicator
    st.subheader("Risk Level")
    st.progress(risk_percentage / 100)

    if risk_percentage < 30:
        st.write("Risk Level: LOW")
    elif risk_percentage < 60:
        st.write("Risk Level: MODERATE")
    else:
        st.write("Risk Level: HIGH")

    # Feature importance (only for tree-based models)
    if hasattr(model, "feature_importances_"):
        st.subheader("Key Risk Factors")
        importance = model.feature_importances_
        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importance
        }).sort_values("Importance", ascending=False).head(5)

        st.bar_chart(importance_df.set_index("Feature"))
    else:
        st.subheader("Model Coefficients")
        coefficients = model.coef_[0]
        coef_df = pd.DataFrame({
            "Feature": feature_names,
            "Weight": np.abs(coefficients)
        }).sort_values("Weight", ascending=False).head(5)

        st.bar_chart(coef_df.set_index("Feature"))

    st.caption("This tool is for educational purposes only. Always consult a medical professional.")
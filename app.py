import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("churn_pipeline.joblib")

# Define mappings (must match your training preprocessing)
gender_map = {"Female": 0, "Male": 1}
yesno_map = {"Yes": 1, "No": 0}
internet_map = {"DSL": 0, "Fiber optic": 1, "No": 2}
contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
payment_map = {
    "Electronic check": 0,
    "Mailed check": 1,
    "Bank transfer (automatic)": 2,
    "Credit card (automatic)": 3
}

st.title("Customer Churn Prediction App")

# Collect user inputs
gender = st.selectbox("Gender", ["Female", "Male"])
SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
Partner = st.selectbox("Partner", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=1)
PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No"])
InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
OnlineSecurity = st.selectbox("Online Security", ["Yes", "No"])
OnlineBackup = st.selectbox("Online Backup", ["Yes", "No"])
DeviceProtection = st.selectbox("Device Protection", ["Yes", "No"])
TechSupport = st.selectbox("Tech Support", ["Yes", "No"])
StreamingTV = st.selectbox("Streaming TV", ["Yes", "No"])
StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No"])
Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
PaymentMethod = st.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
)
MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=50.0)
TotalCharges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=500.0)

# Create DataFrame
df = pd.DataFrame({
    "gender": [gender],
    "SeniorCitizen": [SeniorCitizen],
    "Partner": [Partner],
    "Dependents": [Dependents],
    "tenure": [tenure],
    "PhoneService": [PhoneService],
    "MultipleLines": [MultipleLines],
    "InternetService": [InternetService],
    "OnlineSecurity": [OnlineSecurity],
    "OnlineBackup": [OnlineBackup],
    "DeviceProtection": [DeviceProtection],
    "TechSupport": [TechSupport],
    "StreamingTV": [StreamingTV],
    "StreamingMovies": [StreamingMovies],
    "Contract": [Contract],
    "PaperlessBilling": [PaperlessBilling],
    "PaymentMethod": [PaymentMethod],
    "MonthlyCharges": [MonthlyCharges],
    "TotalCharges": [TotalCharges]
})

# Apply mappings to convert categorical -> numeric
df["gender"] = df["gender"].map(gender_map)
df["Partner"] = df["Partner"].map(yesno_map)
df["Dependents"] = df["Dependents"].map(yesno_map)
df["PhoneService"] = df["PhoneService"].map(yesno_map)
df["MultipleLines"] = df["MultipleLines"].map(yesno_map)
df["OnlineSecurity"] = df["OnlineSecurity"].map(yesno_map)
df["OnlineBackup"] = df["OnlineBackup"].map(yesno_map)
df["DeviceProtection"] = df["DeviceProtection"].map(yesno_map)
df["TechSupport"] = df["TechSupport"].map(yesno_map)
df["StreamingTV"] = df["StreamingTV"].map(yesno_map)
df["StreamingMovies"] = df["StreamingMovies"].map(yesno_map)
df["PaperlessBilling"] = df["PaperlessBilling"].map(yesno_map)

df["InternetService"] = df["InternetService"].map(internet_map)
df["Contract"] = df["Contract"].map(contract_map)
df["PaymentMethod"] = df["PaymentMethod"].map(payment_map)

# Prediction
if st.button("Predict"):
    prediction = model.predict(df)[0]
    if prediction == 1:
        st.error("⚠️ This customer is likely to churn.")
    else:
        st.success("✅ This customer is likely to stay.")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

class ChurnPredictor:
    def __init__(self):
        self.model = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def load_and_preprocess_data(self, filepath):
        """Load and preprocess the customer data"""
        try:
            data = pd.read_csv(filepath)
            # Remove customerID
            if 'customerID' in data.columns:
                data.drop("customerID", axis=1, inplace=True)
            
            # Handle TotalCharges
            data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors='coerce')
            data["TotalCharges"].fillna(data["TotalCharges"].mean(), inplace=True)
            
            # Encode categorical variables
            for col in data.columns:
                if data[col].dtype == 'object':
                    data[col] = self.label_encoder.fit_transform(data[col].astype(str))
            
            self.feature_names = [col for col in data.columns if col != 'Churn']
            return data
        except Exception as e:
            print(f"Error in data preprocessing: {str(e)}")
            raise

# Load the data
data = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")  # Load from data directory

# See first 5 rows
print(data.head())

# Check basic info about columns
print(data.info())

# Check if there are missing values
print(data.isnull().sum())

# Remove customerID column (not useful for prediction)
data.drop("customerID", axis=1, inplace=True)

# Replace spaces in 'TotalCharges' and convert to numbers
data["TotalCharges"] = data["TotalCharges"].replace(" ", np.nan)
data["TotalCharges"] = data["TotalCharges"].astype(float)

# Fill missing values with average
data["TotalCharges"].fillna(data["TotalCharges"].mean(), inplace=True)

# Convert Yes/No and other text columns to numbers
label_encoder = LabelEncoder()
for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = label_encoder.fit_transform(data[col])

X = data.drop("Churn", axis=1)  # all columns except Churn
y = data["Churn"]               # the target (0 = stayed, 1 = left)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)  # Train the model

y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Detailed report
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Feature Importance
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()

import joblib
joblib.dump(model, "churn_pipeline.joblib")

from google.colab import files
files.download("churn_pipeline.joblib")

import joblib

# Load the saved model
model = joblib.load("churn_pipeline.joblib")

print(type(model))        # to check what’s inside
print(model)              # summary of the model

print(model.feature_names_in_)


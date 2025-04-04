# train_models.py

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor, XGBClassifier

# Load dataset
df = pd.read_csv("employee_attrition_and_engagement.csv")

# Drop rows with missing values (can improve this later)
df.dropna(inplace=True)

# Encode categorical variables
label_encoders = {}
for col in ['Department', 'Education']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Select features and targets
X = df[['Age', 'Department', 'Tenure', 'JobSatisfaction', 'WorkHours', 'Education']]
y_performance = df['PerformanceRating']                    # Regression target
y_retention = df['Attrition'].map({'Yes': 1, 'No': 0})     # Classification target

# Train-test split
X_train, X_test, y_perf_train, y_perf_test = train_test_split(X, y_performance, test_size=0.2, random_state=42)
_, _, y_ret_train, y_ret_test = train_test_split(X, y_retention, test_size=0.2, random_state=42)

# Train models
reg = XGBRegressor()
clf = XGBClassifier()

reg.fit(X_train, y_perf_train)
clf.fit(X_train, y_ret_train)

# Save models
import os
os.makedirs("models", exist_ok=True)

with open("models/performance_model.pkl", "wb") as f:
    pickle.dump(reg, f)

with open("models/retention_model.pkl", "wb") as f:
    pickle.dump(clf, f)

with open("models/encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

print("✅ Models trained and saved in 'models/' folder.")


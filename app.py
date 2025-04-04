# app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb

# Load models and encoders
performance_model = pickle.load(open("models/performance_model.pkl", "rb"))
retention_model = pickle.load(open("models/retention_model.pkl", "rb"))
encoders = pickle.load(open("models/encoders.pkl", "rb"))

# App Layout
st.set_page_config(page_title="Employee Predictor", layout="wide")
st.title("🧠 Employee Performance & Retention Predictor")

st.sidebar.header("🔍 Enter Employee Details")

# Manual Input
def user_input_features():
    age = st.sidebar.slider('Age', 18, 60, 30)
    department = st.sidebar.selectbox('Department', ['Sales', 'HR', 'Development', 'Finance', 'Support'])
    tenure = st.sidebar.slider('Years at Company', 0, 20, 3)
    job_satisfaction = st.sidebar.slider('Job Satisfaction (1-5)', 1, 5, 3)
    work_hours = st.sidebar.slider('Average Work Hours/Week', 20, 60, 40)
    education = st.sidebar.selectbox('Education Level', ['Below College', 'College', 'Bachelor', 'Master', 'Doctor'])

    data = {
        'Age': age,
        'Department': department,
        'Tenure': tenure,
        'JobSatisfaction': job_satisfaction,
        'WorkHours': work_hours,
        'Education': education
    }
    return pd.DataFrame([data])

# Collect input
input_df = user_input_features()

# Encode input
for col in input_df.columns:
    if col in encoders:
        input_df[col] = encoders[col].transform(input_df[col])

# Predict
performance_score = performance_model.predict(input_df)[0]
retention_prob = retention_model.predict_proba(input_df)[0][1]
will_leave = "Yes" if retention_prob > 0.5 else "No"

# Show results
st.subheader("📊 Prediction Results")
st.metric("Predicted Performance Score", round(performance_score, 2))
st.metric("Likelihood to Leave", f"{retention_prob * 100:.1f}%")
st.markdown(f"**Will likely leave?** {'🟥 Yes' if will_leave == 'Yes' else '🟩 No'}")

# Show feature importance
def show_feature_importance(model, title):
    booster = model.get_booster()
    importance = booster.get_score(importance_type='gain')
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    st.subheader(title)
    for feat, score in importance.items():
        st.write(f"- **{feat}**: {score:.2f}")

st.divider()
show_feature_importance(performance_model, "📌 Factors Influencing Performance")
show_feature_importance(retention_model, "📌 Factors Influencing Retention")

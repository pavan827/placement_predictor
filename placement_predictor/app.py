"""
Streamlit app — Student Placement Predictor
Run with: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Placement Predictor", page_icon="🎓", layout="centered")

@st.cache_resource
def load_model():
    with open("model/placement_model.pkl", "rb") as f:
        return pickle.load(f)

data = load_model()
model = data["model"]
scaler = data["scaler"]
columns = data["columns"]

st.title("🎓 Student Placement Predictor")
st.caption(f"Model: {data['model_name']} | Test Accuracy: {data['accuracy']*100:.1f}%")
st.write("Enter student details below to predict placement chances.")

col1, col2 = st.columns(2)

with col1:
    cgpa = st.slider("CGPA", 5.0, 10.0, 7.5, 0.1)
    attendance = st.slider("Attendance (%)", 50, 100, 80)
    internships = st.number_input("Internships completed", 0, 5, 1)
    projects = st.number_input("Projects completed", 0, 10, 2)

with col2:
    backlogs = st.number_input("Active Backlogs", 0, 5, 0)
    communication_skill = st.slider("Communication Skill (1-10)", 1, 10, 6)
    coding_score = st.slider("Coding Score (0-100)", 0, 100, 60)
    extra_curricular = st.radio("Active in Extra-curriculars?", ["No", "Yes"])

extra_val = 1 if extra_curricular == "Yes" else 0

if st.button("Predict Placement", type="primary"):
    input_df = pd.DataFrame([[
        cgpa, attendance, internships, projects, backlogs,
        communication_skill, coding_score, extra_val
    ]], columns=columns)

    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0][1]

    st.divider()
    if prediction == 1:
        st.success(f"✅ Likely to be PLACED — Confidence: {proba*100:.1f}%")
    else:
        st.error(f"❌ Likely NOT Placed — Confidence: {(1-proba)*100:.1f}%")

    st.progress(float(proba))
    st.caption("This is a probability estimate from a trained ML model, not a guarantee.")

st.divider()
st.caption("Built with Python, Scikit-learn & Streamlit · College Mini Project")

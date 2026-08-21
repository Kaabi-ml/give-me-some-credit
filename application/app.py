import streamlit as st
import joblib
import numpy as np

# Charger le modèle
model = joblib.load("model.pkl")

# Interface
st.title("🏦 Credit Risk Scoring Tool")
st.write("Assess the default risk of a borrower based on financial profile.")

st.sidebar.header("Client Information")

age = st.sidebar.slider("Age", 18, 80, 35)
debt_ratio = st.sidebar.slider("Debt Ratio (%)", 0, 100, 30)
monthly_income = st.sidebar.number_input("Monthly Income ($)", 0, 50000, 5000)
nb_credits = st.sidebar.number_input("Number of Open Credit Lines", 0, 20, 3)
nb_dependents = st.sidebar.number_input("Number of Dependents", 0, 10, 1)
late_30 = st.sidebar.number_input("Late Payments 30-59 days", 0, 10, 0)
late_60 = st.sidebar.number_input("Late Payments 60-89 days", 0, 10, 0)
late_90 = st.sidebar.number_input("Late Payments 90+ days", 0, 10, 0)
real_estate = st.sidebar.number_input("Number of Real Estate Loans", 0, 10, 0)
revolving = st.sidebar.slider("Revolving Utilization Rate (%)", 0, 100, 20)

if st.button("Assess Credit Risk"):
    features = np.array([[revolving/100, age, late_30,
                          debt_ratio/100, monthly_income,
                          nb_credits, late_90, real_estate,
                          late_60, nb_dependents]])

    proba = model.predict_proba(features)[0][1]

    st.markdown("---")
    st.subheader("Risk Assessment Result")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Default Probability", f"{proba:.1%}")
    with col2:
        if proba < 0.3:
            st.success("✅ LOW RISK")
        elif proba < 0.6:
            st.warning("⚠️ MODERATE RISK")
        else:
            st.error("❌ HIGH RISK")

    # Gauge visuelle
    st.progress(float(proba))
    st.caption(f"Probability of serious delinquency in next 2 years : {proba:.1%}")
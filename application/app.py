import streamlit as st
import joblib
import numpy as np
import shap



# Charger le modèle
model = joblib.load("model.pkl")

# Interface
st.title("Credit Risk Scoring Tool")
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

feature_names = [
    "Revolving Utilization",
    "Age",
    "Late Payments (30–59 days)",
    "Debt Ratio",
    "Monthly Income",
    "Open Credit Lines",
    "Late Payments (90+ days)",
    "Real Estate Loans",
    "Late Payments (60–89 days)",
    "Dependents"
]


best_threshold = 0.471
st.markdown("""
<style>
.risk-circle {
    width: 180px;
    height: 180px;
    border-radius: 50%;
    background: conic-gradient(
        #4CAF50 var(--percentage),
        #2b2f3a var(--percentage)
    );
    display: flex;
    align-items: center;
    justify-content: center;
    margin: auto;
}

.risk-circle::before {
    content: "";
    position: absolute;
    width: 145px;
    height: 145px;
    background: #0e1117;
    border-radius: 50%;
}

.risk-text {
    position: relative;
    z-index: 1;
    text-align: center;
}

.risk-text strong {
    display: block;
    font-size: 32px;
}

.risk-text span {
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

background = joblib.load("shap_background.pkl")

outer_pipeline = model
inner_pipeline = model.named_steps["model"]

scaler = inner_pipeline.named_steps["standardscaler"]
logreg = inner_pipeline.named_steps["logisticregression"]

background_scaled = scaler.transform(background)

explainer = shap.LinearExplainer(
    logreg,
    background_scaled
)


if st.button("Assess Credit Risk"):
    features = np.array([[revolving/100, age, late_30,
                          debt_ratio/100, monthly_income,
                          nb_credits, late_90, real_estate,
                          late_60, nb_dependents]])

    proba = model.predict_proba(features)[0][1]

    st.markdown("---")
    st.subheader("Risk Assessment Result")

    col1, col2= st.columns(2)
    with col1:
        percentage = proba * 100

        st.markdown(
            f"""
            <div class="risk-circle"
                 style="--percentage: {percentage}%;">
                <div class="risk-text">
                    <strong>{percentage:.1f}%</strong>
                    <span>Default probability</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col2:
        if proba >= best_threshold:
            st.error("⚠️ ELEVATED RISK")
        else:
            st.success("✅ LOWER RISK")

    # Gauge visuelle
    st.progress(float(proba))
    st.caption(f"Probability of serious delinquency in next 2 years : {proba:.1%}")



    # risks factors :
    # a. shap calcul
    features_scaled = scaler.transform(features)
    shap_values = explainer(features_scaled)

    contributions = shap_values.values[0]

    features_scaled = scaler.transform(features)

    # b.Interface

    st.markdown("---")

    st.subheader("Why this prediction?")
    st.caption(
        "SHAP shows which features contributed most to this individual prediction."
    )

    importance = sorted(
        zip(feature_names, contributions),
        key=lambda x: abs(x[1]),
        reverse=True
    )



    col_risk, col_protective = st.columns(2)

    positive = [(n, c) for n, c in importance if c > 0][:3]
    negative = [(n, c) for n, c in importance if c < 0][:3]

    with col_risk:
        st.markdown("#### 🔴 Factors increasing risk")

        for name, contribution in positive:
            st.write(f"**{name}**")
            st.caption(f"+{contribution:.3f}")

    with col_protective:
        st.markdown("#### 🟢 Factors reducing risk")

        for name, contribution in negative:
            st.write(f"**{name}**")
            st.caption(f"{contribution:.3f}")

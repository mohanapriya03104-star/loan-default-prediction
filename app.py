import streamlit as st
import pandas as pd
import pickle
st.set_page_config(page_title="Loan Risk Assessor", layout="wide")
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #E6E6FA, #D8BFD8);
    color: #2E0854;
}

h1, h2, h3 {
    color: #4B0082;
}

input, .stNumberInput input {
    background-color: #F3E8FF;
    color: #4B0082;
    border-radius: 8px;
    border: 1px solid #C084FC;
}

.stSelectbox div {
    background-color: #F3E8FF;
    color: #4B0082;
}

.stButton>button {
    background-color: #7C3AED;
    color: white;
    border-radius: 10px;
    height: 50px;
    width: 100%;
    font-size: 18px;
}

.stButton>button:hover {
    background-color: #5B21B6;
    transition: 0.3s;
}

.block-container {
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)
model = pickle.load(open("loan_model.pkl", "rb"))
FEATURES = pickle.load(open("features.pkl", "rb"))
st.markdown("## 💜 Loan Risk Assessor")
st.caption("AI-powered loan approval system")
name = st.text_input("👤 Full Name")
st.markdown("### 💰 Financial Details")
col1, col2, col3 = st.columns(3)
with col1:
    age = st.number_input("Age", 18, 70, 25)
    income = st.number_input("Annual Income (₹)", 1000, 1000000, 50000)
with col2:
    loan_amount = st.number_input("Loan Amount (₹)", 1000, 500000, 100000)
    loan_term = st.number_input("Loan Term (months)", 6, 60, 12)
with col3:
    credit_score = st.number_input("Credit Score", 300, 900, 650)
    interest_rate = st.number_input("Interest Rate (%)", 1, 30, 10)
months_employed = st.number_input("Months Employed", 0, 120, 12)
credit_lines = st.number_input("Number of Credit Lines", 0, 20, 2)
dti_ratio = st.slider("DTI Ratio", 0.0, 1.0, 0.3)
education = {"Graduate":0,"Not Graduate":1,"Other":2}[st.selectbox("Education",["Graduate","Not Graduate","Other"])]
employment = {"Salaried":0,"Self-Employed":1,"Unemployed":2}[st.selectbox("Employment",["Salaried","Self-Employed","Unemployed"])]
marital = {"Single":0,"Married":1}[st.selectbox("Marital Status",["Single","Married"])]
mortgage = {"No":0,"Yes":1}[st.selectbox("Mortgage",["No","Yes"])]
dependents = {"No":0,"Yes":1}[st.selectbox("Dependents",["No","Yes"])]
cosigner = {"No":0,"Yes":1}[st.selectbox("Co-Signer",["No","Yes"])]
purpose = {"Personal":0,"Business":1,"Education":2,"Other":3}[st.selectbox("Loan Purpose",["Personal","Business","Education","Other"])]
if st.button("🚀 Predict Loan Status"):
    input_df = pd.DataFrame([[
        age, income, loan_amount, credit_score, months_employed,
        credit_lines, interest_rate, loan_term, dti_ratio,
        education, employment, marital, mortgage,
        dependents, purpose, cosigner
    ]], columns=FEATURES)
    prob = model.predict_proba(input_df)[0][1]
    st.markdown("---")
    st.subheader(f"📊 Result for {name if name else 'Applicant'}")
    st.progress(int(prob * 100))

    if prob > 0.7:
        st.error("🔴 High Risk - Loan Rejected")
    elif prob > 0.4:
        st.warning("🟡 Medium Risk - Needs Review")
    else:
        st.success("🟢 Low Risk - Loan Approved")

    st.write(f"### Risk Score: {round(prob*100,2)}%")

import streamlit as st
import numpy as np
import pickle

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="HDI Predictor", layout="centered")

# -------------------------------
# LOAD MODEL
# -------------------------------
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# -------------------------------
# TITLE
# -------------------------------
st.title("🌍 Human Development Index Predictor")
st.markdown("### 📊 Decision Support Tool for Policy Analysis")

st.write("Simulate how changes in **health, education, and income** impact a country's development.")

st.markdown("---")

# -------------------------------
# INPUTS
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    life_exp = st.number_input("🧬 Life Expectancy (years)", 40.0, 90.0, 70.0)
    expected_school = st.number_input("📚 Expected Years of Schooling", 0.0, 25.0, 12.0)

with col2:
    mean_school = st.number_input("🎓 Mean Years of Schooling", 0.0, 20.0, 8.0)
    gni = st.number_input("💰 GNI per Capita (PPP $)", 100.0, 150000.0, 10000.0)

st.markdown("---")

# -------------------------------
# PREDICTION
# -------------------------------
if st.button("🔍 Predict HDI"):

    input_data = np.array([[life_exp, expected_school, mean_school, gni]])
    prediction = model.predict(input_data)[0]

    # -------------------------------
    # RESULT
    # -------------------------------
    st.subheader("📈 Prediction Result")
    st.success(f"Predicted HDI: **{round(prediction, 3)}**")

    # Progress bar
    st.progress(min(max(prediction, 0.0), 1.0))

    # -------------------------------
    # CATEGORY + POLICY INSIGHT
    # -------------------------------
    st.subheader("📊 Development Category")

    if prediction >= 0.8:
        st.success("🌟 Very High Human Development")
        st.write("Policy Focus: Maintain growth through **innovation, sustainability, and advanced education systems**.")

    elif prediction >= 0.7:
        st.info("📈 High Human Development")
        st.write("Policy Focus: Improve **income equality and quality of education** to reach very high development.")

    elif prediction >= 0.55:
        st.warning("⚖️ Medium Human Development")
        st.write("Policy Focus: Invest in **education access, healthcare infrastructure, and economic growth**.")

    else:
        st.error("⚠️ Low Human Development")
        st.write("Policy Focus: Urgent investment in **basic health, primary education, and income generation**.")

    # -------------------------------
    # EXTRA BUSINESS INSIGHT
    # -------------------------------
    st.markdown("### 🧠 Key Insight")
    st.write(
        "HDI is strongly influenced by **education and income levels**. "
        "Long-term investment in human capital can significantly improve development outcomes."
    )

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.caption("🎓 MBA ABA Project | HDI Prediction Model | Data Source: UNDP")
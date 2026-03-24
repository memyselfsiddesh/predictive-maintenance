import streamlit as st
import sys
import os

sys.path.append(os.path.abspath("model"))

from maintenance_model import model_object
st.set_page_config(page_title="Predictive Maintenance", layout="centered")

st.title("🔧 Predictive Maintenance System")
st.subheader("AI + ML based Industrial Failure Prediction")

st.info("Enter machine sensor values to predict failure risk")

# Input fields
air_temp = st.number_input("Air Temperature", min_value=0.0)
process_temp = st.number_input("Process Temperature", min_value=0.0)
speed = st.number_input("Rotational Speed", min_value=0.0)
torque = st.number_input("Torque", min_value=0.0)
wear = st.number_input("Tool Wear", min_value=0.0)

# Prediction
if st.button("Predict"):

    input_data = [[air_temp, process_temp, speed, torque, wear]]

    prediction = model_object.predict(input_data)
    rule_output = model_object.rule_engine(air_temp, process_temp, torque, wear)

    st.subheader("🔍 Results")

    if prediction[0] == 1:
        st.error("⚠️ Machine Failure Likely")
    else:
        st.success("✅ Machine Operating Normally")

    st.write("### AI Rule-Based Insight:")
    st.info(rule_output)

    st.write("### Raw Prediction Value:")
    st.write(prediction[0])
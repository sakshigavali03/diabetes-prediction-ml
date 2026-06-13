import streamlit as st
import numpy as np
import pickle

# Load trained model
model = pickle.load(open('diabetes_model.pkl', 'rb'))

st.title("Diabetes Prediction System")

pregnancies = st.number_input("Pregnancies", min_value=0)
glucose = st.number_input("Glucose Level", min_value=0)
blood_pressure = st.number_input("Blood Pressure", min_value=0)
skin_thickness = st.number_input("Skin Thickness", min_value=0)
insulin = st.number_input("Insulin", min_value=0)
bmi = st.number_input("BMI", min_value=0.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)
age = st.number_input("Age", min_value=1)

if st.button("Predict"):
    input_data = np.array([[pregnancies, glucose, blood_pressure,
                            skin_thickness, insulin, bmi,
                            dpf, age]])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("The person is likely to have Diabetes.")
    else:
        st.success("The person is not likely to have Diabetes.")

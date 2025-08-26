
import streamlit as st
import pickle
import numpy as np

# Load the saved SVM model
with open("svm_model_state26.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🍷 Wine Quality Prediction App By Dewansh01")

st.write("Enter the wine characteristics below to predict its quality.")

# Input fields for each feature
fixed_acidity = st.number_input("Fixed Acidity", min_value=0.0, step=0.1)
volatile_acidity = st.number_input("Volatile Acidity", min_value=0.0, step=0.01)
citric_acid = st.number_input("Citric Acid", min_value=0.0, step=0.01)
residual_sugar = st.number_input("Residual Sugar", min_value=0.0, step=0.1)
chlorides = st.number_input("Chlorides", min_value=0.0, step=0.001)
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", min_value=0.0, step=1.0)
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", min_value=0.0, step=1.0)
density = st.number_input("Density", min_value=0.0, step=0.0001, format="%.5f")
pH = st.number_input("pH", min_value=0.0, step=0.01)
sulphates = st.number_input("Sulphates", min_value=0.0, step=0.01)
alcohol = st.number_input("Alcohol", min_value=0.0, step=0.1)

# Dropdown for type (assuming Red/White wine)
wine_type = st.selectbox("Type of Wine", ["Red", "White"])
wine_type_encoded = 0 if wine_type == "Red" else 1

# Predict button
if st.button("Predict Quality"):
    features = np.array([[fixed_acidity, volatile_acidity, citric_acid,
                          residual_sugar, chlorides, free_sulfur_dioxide,
                          total_sulfur_dioxide, density, pH, sulphates,
                          alcohol, wine_type_encoded]])
    
    prediction = model.predict(features)
    st.success(f"Predicted Wine Quality: {prediction[0]}")

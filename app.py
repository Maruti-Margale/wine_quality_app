import streamlit as st
import pickle
import numpy as np
import os
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pandas as pd

st.title("🍷 Wine Quality Prediction App By Maruti 😎")
st.write("Enter the wine characteristics below to predict its quality.")

# Path for the saved model
MODEL_FILE = os.path.join(os.path.dirname(__file__), "svm_model_state26.pkl")

# ✅ Function to train and save the model if not available
def train_and_save_model():
    st.info("Training new SVM model... Please wait!")
    # Download Wine Quality dataset (Red and White combined)
    url_red = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    url_white = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"

    # Load datasets
    red = pd.read_csv(url_red, sep=";")
    red["type"] = 0  # Red wine
    white = pd.read_csv(url_white, sep=";")
    white["type"] = 1  # White wine
    data = pd.concat([red, white], axis=0)

    # Features and target
    X = data.drop(columns=["quality"])
    y = data["quality"]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train SVM
    model = svm.SVC(kernel="rbf")
    model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    st.success(f"Model trained with accuracy: {acc:.2f}")

    # Save model and scaler
    with open(MODEL_FILE, "wb") as f:
        pickle.dump((model, scaler), f)

    return model, scaler

# ✅ Load or Train Model
if os.path.exists(MODEL_FILE):
    with open(MODEL_FILE, "rb") as f:
        model, scaler = pickle.load(f)
else:
    model, scaler = train_and_save_model()

# ✅ Input fields
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

wine_type = st.selectbox("Type of Wine", ["Red", "White"])
wine_type_encoded = 0 if wine_type == "Red" else 1

# ✅ Prediction
if st.button("Predict Quality"):
    features = np.array([[fixed_acidity, volatile_acidity, citric_acid,
                          residual_sugar, chlorides, free_sulfur_dioxide,
                          total_sulfur_dioxide, density, pH, sulphates,
                          alcohol, wine_type_encoded]])
    
    # Scale input before prediction
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)

    st.success(f"Predicted Wine Quality: {prediction[0]}")

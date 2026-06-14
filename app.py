import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_excel("Crop_recommendation.csv.xlsx")

# Features and target
X = df.drop("label", axis=1)
y = df["label"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Heading
st.title("🌾 Crop Recommendation System")
st.write("Enter soil and weather values")

# Inputs
N = st.number_input("Nitrogen (N)", min_value=0, step=10)
P = st.number_input("Phosphorus (P)", min_value=0, step=10)
K = st.number_input("Potassium (K)", min_value=0, step=10)

temperature = st.number_input(
    "Temperature", min_value=0.0, step=1.0
)

humidity = st.number_input(
    "Humidity", min_value=0.0, step=1.0
)

ph = st.number_input(
    "pH", min_value=0.0, step=0.1
)

rainfall = st.number_input(
    "Rainfall", min_value=0.0, step=10.0
)

# Button
if st.button("Predict Crop"):

    data = pd.DataFrame(
        [[N, P, K, temperature, humidity, ph, rainfall]],
        columns=[
            "N",
            "P",
            "K",
            "temperature",
            "humidity",
            "ph",
            "rainfall"
        ]
    )

    prediction = model.predict(data)

    st.success(
        "Recommended Crop: " + prediction[0]
    )
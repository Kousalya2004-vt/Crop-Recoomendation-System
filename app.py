import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(
    page_title="Crop Recommendation",
    page_icon="🌾",
    layout="wide"
)
st.markdown("""
<style>
.stApp {
    background-image: url("https://images.unsplash.com/photo-1464226184884-fa280b87c399");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}
</style>
""", unsafe_allow_html=True)

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
st.image("farmer.jpg", width=500)
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


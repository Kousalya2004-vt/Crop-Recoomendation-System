import streamlit as st
import pandas as pd
import requests
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
    background: linear-gradient(to right, #e8f5e9, #c8e6c9);
}

h1 {
    color: #1b5e20;
    text-align: center;
}

[data-testid="stVerticalBlock"] {
    background-color: rgba(255,255,255,0.85);
    padding: 20px;
    border-radius: 15px;
}

.stButton>button {
    background-color: #2e7d32;
    color: white;
    border-radius: 10px;
    font-size: 18px;
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


st.title("🌾 AI Crop Recommendation System")
st.markdown("### Smart Agriculture using Machine Learning and Real-Time Weather Data")
st.write("Enter soil and weather values")

city = st.text_input("Enter City Name")

temperature = None
humidity = None

if st.button("Fetch Weather"):

    API_KEY = "80a1816fc6a52aed12448e0c2f5e1887"

    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"

    response = requests.get(url)

    if response.status_code == 200:
        weather = response.json()

        temperature = weather["main"]["temp"]
        humidity = weather["main"]["humidity"]

        st.success(f"Temperature: {temperature} °C")
        st.success(f"Humidity: {humidity} %")

# Inputs
N = st.number_input("Nitrogen (N)", min_value=0, step=10)
P = st.number_input("Phosphorus (P)", min_value=0, step=10)
K = st.number_input("Potassium (K)", min_value=0, step=10)
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
    st.success(f"Recommended Crop: {prediction[0].upper()}")


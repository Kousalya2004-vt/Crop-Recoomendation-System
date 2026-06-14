import pandas as pd
import tkinter as tk
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_excel("Crop_recommendation.csv.xlsx")

# Features and target
X = df.drop("label", axis=1)
y = df["label"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prediction function
def predict_crop():
    try:
        N = float(n_entry.get())
        P = float(p_entry.get())
        K = float(k_entry.get())
        temperature = float(temp_entry.get())
        humidity = float(hum_entry.get())
        ph = float(ph_entry.get())
        rainfall = float(rain_entry.get())

        data = pd.DataFrame(
            [[N, P, K, temperature, humidity, ph, rainfall]],
            columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
        )

        prediction = model.predict(data)

        result_label.config(
            text="Recommended Crop: " + prediction[0]
        )

    except:
        result_label.config(text="Please enter valid values")

# GUI
window = tk.Tk()
window.title("Crop Recommendation System")
window.geometry("400x500")

tk.Label(window, text="Nitrogen (N)").pack()
n_entry = tk.Entry(window)
n_entry.pack()

tk.Label(window, text="Phosphorus (P)").pack()
p_entry = tk.Entry(window)
p_entry.pack()

tk.Label(window, text="Potassium (K)").pack()
k_entry = tk.Entry(window)
k_entry.pack()

tk.Label(window, text="Temperature").pack()
temp_entry = tk.Entry(window)
temp_entry.pack()

tk.Label(window, text="Humidity").pack()
hum_entry = tk.Entry(window)
hum_entry.pack()

tk.Label(window, text="pH").pack()
ph_entry = tk.Entry(window)
ph_entry.pack()

tk.Label(window, text="Rainfall").pack()
rain_entry = tk.Entry(window)
rain_entry.pack()

tk.Button(window, text="Predict Crop", command=predict_crop).pack(pady=10)

result_label = tk.Label(window, text="", font=("Arial", 12))
result_label.pack(pady=20)

window.mainloop()
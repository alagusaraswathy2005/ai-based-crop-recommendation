import streamlit as st
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier
import google.generativeai as genai

import os
from dotenv import load_dotenv

load_dotenv()

WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
llm_model = genai.GenerativeModel("gemini-3.1-flash-lite-preview")

# ------------------ TITLE ------------------
st.title(" Smart Crop Recommendation System")

# ------------------ LOAD DATA ------------------
data = pd.read_csv("Crop_recommendation.csv")
data = data.dropna().drop_duplicates()

# ------------------ TRAIN MODEL ------------------
X = data[['N', 'P', 'K', 'temperature', 'humidity', 'rainfall']]
y = data['label']

model = RandomForestClassifier()
model.fit(X, y)

st.success("✅ Model Ready!")

# ------------------ WEATHER FUNCTION ------------------
def get_weather(city):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
    response = requests.get(url)
    data = response.json()

    if data["cod"] != 200:
        return None

    temp = data["main"]["temp"]
    humidity = data["main"]["humidity"]

    # Rainfall (may not always be present)
    rainfall = data.get("rain", {}).get("1h")

    # fallback value
    if rainfall is None:
        rainfall = 100

    return temp, humidity, rainfall

# ------------------ GEMINI EXPLANATION ------------------
def get_explanation(crop, N, P, K, temp, humidity, rainfall):
    prompt = f"""
    You are an agriculture expert.

    Crop: {crop}
    Nitrogen: {N}
    Phosphorus: {P}
    Potassium: {K}
    Temperature: {temp}°C
    Humidity: {humidity}%
    Rainfall: {rainfall} mm

    Explain in simple English why this crop is suitable.
    """

    try:
        response = llm_model.generate_content(prompt)

        # ✅ Debug print (very important)
        print(response)

        # ✅ Safe return
        if hasattr(response, "text") and response.text:
            return response.text
        else:
            return "⚠️ Explanation not generated properly"

    except Exception as e:
        return f"⚠️ Error: {str(e)}"

# ------------------ USER INPUT ------------------
st.subheader(" Enter Location")
city = st.text_input("City Name")

st.subheader("🌱 Enter Soil Nutrients")

N = st.text_input("Nitrogen (N)")
P = st.text_input("Phosphorus (P)")
K = st.text_input("Potassium (K)")

# ------------------ PREDICT ------------------
if st.button(" Predict Crop"):

    if not city or not N or not P or not K:
        st.error(" Please fill all fields")
    else:
        try:
            # Convert inputs
            N = float(N)
            P = float(P)
            K = float(K)

            weather = get_weather(city)

            if weather:
                temperature, humidity, rainfall = weather

                # Model prediction
                input_data = [[N, P, K, temperature, humidity, rainfall]]

                prediction = model.predict(input_data)[0]
                # probability = model.predict_proba(input_data).max()

                # Output
                st.success(f" Recommended Crop: {prediction}")

                st.write(f" Temperature: {temperature} °C")
                st.write(f" Humidity: {humidity} %")
                st.write(f" Rainfall: {rainfall} mm")

                # Gemini explanation
                explanation = get_explanation(
                    prediction, N, P, K, temperature, humidity, rainfall
                )

                st.subheader("🤖 Why this crop?")
                st.write(explanation)

            else:
                st.error(" Invalid city name")

        except:
            st.error(" Enter valid numeric values")
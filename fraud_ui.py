import streamlit as st
import joblib
import pandas as pd
from pymongo import MongoClient
from streamlit_lottie import st_lottie
import requests
import matplotlib.pyplot as plt

# ===== PAGE CONFIG =====
st.set_page_config(page_title="Fraud Detection Login", page_icon="🔐", layout="centered")

# ===== MONGODB CONNECTION =====
client = MongoClient("mongodb://localhost:27017/")
db = client["fraud_detection"]
users_collection = db["users"]
transactions = db["transactions"]

# ===== Load model and encoders =====
model = joblib.load("fraud_model.pkl")
le_country = joblib.load("le_country.pkl")
le_ip_country = joblib.load("le_ip_country.pkl")

# ===== LOTTIE ANIMATION FUNCTION =====
def load_lottie_url(url):
    try:
        r = requests.get(url)
        if r.status_code == 200:
            return r.json()
    except:
        return None

# Load a working Lottie animation
lottie_fraud = load_lottie_url("https://assets9.lottiefiles.com/packages/lf20_jcikwtux.json")

# ===== IP to country mapping =====
def get_country_from_ip(ip):
    if ip.startswith("49."): return "India"
    elif ip.startswith("3."): return "USA"
    elif ip.startswith("5."): return "Russia"
    elif ip.startswith("102."): return "Nigeria"
    elif ip.startswith("14."): return "Vietnam"
    elif ip.startswith("192."): return "UK"
    elif ip.startswith("203."): return "Russia"
    elif ip.startswith("198."): return "USA"
    elif ip.startswith("36."): return "China"
    elif ip.startswith("45."): return "Nigeria"
    else: return "Unknown"

# ===== Session Init =====
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ===== LOGIN PAGE =====
if not st.session_state.logged_in:
    st.markdown("## 🔐 Welcome to Fraud Detection Login")

    if lottie_fraud:
        st_lottie(lottie_fraud, height=300, key="fraud")
    else:
        st.warning("⚠️ Animation could not load. Please check your connection.")

    auth_mode = st.radio("Choose an option:", ["Login", "Sign Up"])
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if auth_mode == "Sign Up":
        if st.button("Create Account"):
            if users_collection.find_one({"username": username}):
                st.error("🚫 Username already exists.")
            else:
                users_collection.insert_one({"username": username, "password": password})
                st.success("✅ Account created! Please login now.")

    elif auth_mode == "Login":
        if st.button("Login"):
            user = users_collection.find_one({"username": username, "password": password})
            if user:
                st.session_state.logged_in = True
                st.success(f"🎉 Welcome, {username}!")
                st.rerun()  # ✅ use st.rerun() instead of experimental
            else:
                st.error("❌ Invalid credentials.")

# ===== MAIN APP AFTER LOGIN =====
if st.session_state.logged_in:

    # ===== Custom Header Style =====
    st.markdown(
        """
        <style>
            .block-container {
                max-width: 800px;
                margin: auto;
                padding-top: 1rem;
            }
            .title-text {
                text-align: center;
                font-size: 32px;
                font-weight: 700;
                margin-bottom: 1.5rem;
            }
        </style>
        <div class='title-text'>💳 Real-Time Fraud Detection System</div>
        """,
        unsafe_allow_html=True
    )

    # ===== TABS =====
    tab1, tab2, tab3 = st.tabs(["🔍 Predict Fraud", "📋 View Stored Logs", "📊 Visual Analysis"])

    # === TAB 1: Prediction ===
    with tab1:
        st.markdown("### 🧾 Prediction Form")
        ip_address = st.text_input("Enter IP Address (e.g., 49.36.77.18)")
        user_country = st.selectbox("Select User-Declared Country", sorted(le_country.classes_))
        amount = st.number_input("Enter Transaction Amount (₹)", min_value=1, step=1, format="%d")

        if st.button("Predict"):
            country_from_ip = get_country_from_ip(ip_address)
            st.write(f"📍 IP maps to country: **{country_from_ip}**")

            try:
                encoded_country = le_country.transform([user_country])[0]
                encoded_ip_country = le_ip_country.transform([country_from_ip])[0]
            except:
                st.error("❌ Country not found in training data.")
                st.stop()

            input_df = pd.DataFrame([[encoded_country, encoded_ip_country, amount]],
                                    columns=["country", "country_from_ip", "amount"])
            
            prediction = model.predict(input_df)[0]
            result = "🚨 FRAUD" if prediction == 1 else "✅ LEGIT"
            st.success(f"Prediction: {result}")

            # Save to DB
            transactions.insert_one({
                "ip_address": ip_address,
                "user_country": user_country,
                "country_from_ip": country_from_ip,
                "amount": amount,
                "prediction": result
            })
            st.info("📦 Transaction saved to MongoDB")

    # === TAB 2: Logs ===
    with tab2:
        st.markdown("### 📋 Stored Transactions")
        all_data = list(transactions.find({}, {"_id": 0}))
        if all_data:
            df_logs = pd.DataFrame(all_data)
            st.dataframe(df_logs, use_container_width=True)
            csv = df_logs.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Logs as CSV", data=csv, file_name="fraud_logs.csv", mime='text/csv')
        else:
            st.warning("No transactions logged yet.")

    # === TAB 3: Chart ===
    with tab3:
        st.markdown("### 📊 Fraud vs Legit Chart")
        all_data = list(transactions.find({}, {"_id": 0}))
        if all_data:
            df_logs = pd.DataFrame(all_data)
            count_data = df_logs['prediction'].value_counts()
            labels = count_data.index.tolist()
            values = count_data.values.tolist()

            fig, ax = plt.subplots()
            ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90, colors=['red', 'green'])
            ax.axis('equal')
            st.pyplot(fig)
        else:
            st.info("No data available to show chart.")

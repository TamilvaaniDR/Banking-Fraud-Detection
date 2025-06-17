import streamlit as st
import pandas as pd
from pymongo import MongoClient
from streamlit_lottie import st_lottie
import requests
import matplotlib.pyplot as plt
from datetime import datetime

# ===== PAGE CONFIG =====
st.set_page_config(page_title="Fraud Detection Login", page_icon="🔐", layout="centered")

# ===== MONGODB CONNECTION =====
client = MongoClient("mongodb://localhost:27017/")
db = client["fraud_detection"]
users_collection = db["users"]
transactions = db["transactions"]

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

# ===== Rule-based fraud detection =====
def check_fraud(ip_address, country, amount, transaction_time):
    country_from_ip = get_country_from_ip(ip_address)

    if country != country_from_ip:
        return "FRAUD"

    hour = datetime.strptime(transaction_time, "%H:%M").hour
    if hour < 6 or hour > 22:
        return "FRAUD"

    if amount > 5000:
        return "FRAUD"

    return "NON-FRAUD"

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
                st.rerun()
            else:
                st.error("❌ Invalid credentials.")

# ===== MAIN APP AFTER LOGIN =====
if st.session_state.logged_in:

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

    tab1, tab2, tab3 = st.tabs(["🔍 Predict Fraud", "📋 View Stored Logs", "📊 Visual Analysis"])

    with tab1:
        st.markdown("### 🧾 Prediction Form")
        ip_address = st.text_input("Enter IP Address (e.g., 49.36.77.18)")
        user_country = st.text_input("Enter Country", "India")
        amount = st.number_input("Enter Transaction Amount (₹)", min_value=1, step=1, format="%d")
        time_input = st.time_input("Enter Transaction Time (HH:MM)", value=datetime.now())

        if st.button("Check Fraud"):
            transaction_time = time_input.strftime("%H:%M")
            country_from_ip = get_country_from_ip(ip_address)
            result = check_fraud(ip_address, user_country, amount, transaction_time)
            st.write(f"📍 IP maps to country: **{country_from_ip}**")
            st.success(f"Prediction: {'🚨 FRAUD' if result == 'FRAUD' else '✅ LEGIT'}")

            transactions.insert_one({
                "ip_address": ip_address,
                "user_country": user_country,
                "country_from_ip": country_from_ip,
                "amount": amount,
                "transaction_time": transaction_time,
                "prediction": result
            })
            st.info("📦 Transaction saved to MongoDB")

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

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from datetime import datetime

# ============================
# 1. TELEGRAM CONFIG
# ============================
BOT_TOKEN = st.secrets["BOT_TOKEN"]
CHAT_ID = st.secrets["CHAT_ID"]


def send_telegram(message):
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        payload = {"chat_id": CHAT_ID, "text": message}
        requests.post(url, data=payload, timeout=5)
    except:
        pass

# ============================
# 2. WEB INTERFACE (STREAMLIT)
# ============================
st.set_page_config(page_title="Kolkata Quant", layout="centered")

st.title("🚀 Kolkata Quant: NSE Predictor")
st.markdown("Enter stock details to get ML predictions and signals.")

# Sidebar for User Input
st.sidebar.header("User Settings")
stock_code = st.sidebar.text_input("NSE Stock Code (e.g. SBIN)", "RELIANCE").upper()
capital = st.sidebar.number_input("Total Capital (₹)", value=100000)
risk_percent = st.sidebar.slider("Risk per Trade (%)", 0.5, 5.0, 2.0)

ticker = stock_code + ".NS"

if st.button("Run ML Analysis"):
    with st.spinner(f"Analyzing {ticker}..."):
        try:
            # Data Download
            df = yf.download(ticker, period="1y", interval="1d", progress=False)
            
            if df.empty:
                st.error("Invalid stock code! Please use standard NSE codes like TCS, INFY, etc.")
            else:
                # Feature Engineering (Your exact logic)
                df["Return"] = df["Close"].pct_change()
                df["MA20"] = df["Close"].rolling(20).mean()
                df["MA50"] = df["Close"].rolling(50).mean()
                df["Volatility"] = df["Return"].rolling(20).std()
                df.dropna(inplace=True)

                features = ["MA20", "MA50", "Return", "Volatility"]
                X = df[features]
                y = df["Close"].shift(-1)
                X_clean, y_clean = X.iloc[:-1], y.iloc[:-1]

                # ML Training
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_clean)
                model = LinearRegression().fit(X_scaled, y_clean)

                # Prediction
                last_features = scaler.transform(X.iloc[-1].values.reshape(1, -1))
                predicted_price = model.predict(last_features).item()
                last_price = float(df["Close"].iloc[-1])

                # Signal Logic
                change = (predicted_price - last_price) / last_price
                if change > 0.003:
                    signal, color = "BUY [UP]", "green"
                elif change < -0.003:
                    signal, color = "SELL [DOWN]", "red"
                else:
                    signal, color = "HOLD [STABLE]", "gray"

                # Risk Management
                stop_loss = last_price * 0.98
                risk_amount = capital * (risk_percent / 100)
                qty = int(risk_amount / abs(last_price - stop_loss))

                # Display Results
                st.subheader(f"Analysis for {ticker}")
                col1, col2, col3 = st.columns(3)
                col1.metric("Current Price", f"₹{last_price:.2f}")
                col2.metric("Predicted Price", f"₹{predicted_price:.2f}", f"{change*100:.2f}%")
                col3.metric("Trade Signal", signal)

                st.info(f"**Action:** Recommended Quantity: **{qty}** | Stop Loss: **₹{stop_loss:.2f}**")

                # Telegram Alert
                if signal != "HOLD [STABLE]":
                    msg = f"WEB APP ALERT:\n{ticker}\nSignal: {signal}\nPrice: ₹{last_price:.2f}\nQty: {qty}"
                    send_telegram(msg)

                # Show Chart
                st.subheader("Visual Price Trend")
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(df["Close"], label="Close Price", color='blue')
                ax.plot(df["MA20"], label="20d Moving Avg", color='orange')
                ax.plot(df["MA50"], label="50d Moving Avg", color='green')
                ax.legend()
                st.pyplot(fig)

        except Exception as e:

            st.error(f"Analysis Error: {e}")

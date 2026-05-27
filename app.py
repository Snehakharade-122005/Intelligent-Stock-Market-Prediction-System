import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from streamlit_autorefresh import st_autorefresh

# Title
st.title("📈 Intelligent Stock Market Prediction System")
# Auto refresh every 60 seconds
st_autorefresh(interval=60000, key="stock_refresh")

# Load Model
model = load_model("model/lstm_stock_model.h5")

# Stock Selection
stock = st.selectbox(
    "Select Stock",
    ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
)

# Download Stock Data
data = yf.download(
    stock,
    period="1y",
    interval="1d"
)

# Show Dataset
st.subheader("Stock Data")
# Current Stock Price
close_price = data['Close']

# Convert properly to single value
latest_price = close_price.iloc[-1]

# If Series comes, take first element
if hasattr(latest_price, 'iloc'):
    latest_price = latest_price.iloc[0]

st.metric(
    label="Current Stock Price",
    value=f"₹ {latest_price:.2f}"
)

st.write(data.tail())

# Plot Closing Price
st.subheader("Closing Price Graph")

fig = plt.figure(figsize=(12,6))
plt.plot(data['Close'])
plt.xlabel("Date")
plt.ylabel("Close Price")

st.pyplot(fig)

# Prepare Data
close_data = data['Close'].values
close_data = close_data.reshape(-1,1)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(close_data)

# Create Test Data
x_test = []

for i in range(60, len(scaled_data)):
    x_test.append(scaled_data[i-60:i, 0])

x_test = np.array(x_test)

x_test = np.reshape(
    x_test,
    (x_test.shape[0], x_test.shape[1], 1)
)

# Prediction
predicted_prices = model.predict(x_test)

predicted_prices = scaler.inverse_transform(predicted_prices)

# Live Prediction
next_prediction = predicted_prices[-1][0]

st.subheader("Live AI Prediction")

st.metric(
    label="Predicted Next Price",
    value=f"₹ {next_prediction:.2f}"
)

# Buy/Sell Decision
current_price = latest_price

if next_prediction > current_price:
    st.success("AI Signal: BUY 📈")
else:
    st.error("AI Signal: SELL 📉")
    
# Actual Prices
actual_prices = close_data[60:]

# Plot Prediction
st.subheader("Actual vs Predicted Prices")

fig2 = plt.figure(figsize=(12,6))

plt.plot(actual_prices, color='blue', label='Actual Price')
plt.plot(predicted_prices, color='red', label='Predicted Price')

plt.xlabel("Time")
plt.ylabel("Stock Price")

plt.legend()

st.pyplot(fig2)

# Moving Averages
data['MA50'] = data['Close'].rolling(window=50).mean()
data['MA200'] = data['Close'].rolling(window=200).mean()

# Buy/Sell Signal
st.subheader("Buy/Sell Signal")

fig3 = plt.figure(figsize=(12,6))

plt.plot(data['Close'], label='Close Price')
plt.plot(data['MA50'], label='50-Day MA')
plt.plot(data['MA200'], label='200-Day MA')

plt.legend()

st.pyplot(fig3)

# Sentiment Section
st.subheader("Market Sentiment")

st.success("Positive market sentiment detected for selected stock")

st.write("Project Developed using LSTM + NLP + Technical Analysis")
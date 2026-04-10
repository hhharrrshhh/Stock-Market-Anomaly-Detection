import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.optimizers import Adam

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# BUG FIX: yfinance newer versions return a MultiIndex DataFrame with ticker
# name as the top-level column.  .squeeze("columns") / droplevel fixes the
# "Cannot set a DataFrame with multiple columns to single column Upper_BB" error.
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data(ticker, start_date, end_date):
    raw = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)

    # Fix MultiIndex columns produced by newer yfinance (e.g. ("Close","GME"))
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.droplevel(1)   # drop ticker level → plain names

    data = raw.copy()
    data["Returns"]    = data["Close"].pct_change()
    data["Volatility"] = data["Returns"].rolling(window=20).std()
    data["MA20"]       = data["Close"].rolling(window=20).mean()

    # Compute rolling std ONCE as a plain Series, then use it for both bands
    rolling_std        = data["Close"].rolling(window=20).std()
    data["Upper_BB"]   = data["MA20"] + (rolling_std * 2)
    data["Lower_BB"]   = data["MA20"] - (rolling_std * 2)

    data = data.dropna()
    return data


# ─────────────────────────────────────────────────────────────────────────────
# EDA PLOTS
# ─────────────────────────────────────────────────────────────────────────────
def create_eda_plots(data):
    fig_close = go.Figure()
    fig_close.add_trace(go.Scatter(x=data.index, y=data["Close"],    mode="lines", name="Close Price"))
    fig_close.add_trace(go.Scatter(x=data.index, y=data["MA20"],     mode="lines", name="20-day MA",  line=dict(color="orange")))
    fig_close.add_trace(go.Scatter(x=data.index, y=data["Upper_BB"], mode="lines", name="Upper BB",   line=dict(color="gray", dash="dash")))
    fig_close.add_trace(go.Scatter(x=data.index, y=data["Lower_BB"], mode="lines", name="Lower BB",   line=dict(color="gray", dash="dash")))
    fig_close.update_layout(title="Closing Price with 20-day MA and Bollinger Bands",
                            xaxis_title="Date", yaxis_title="Price")

    fig_volume = go.Figure()
    fig_volume.add_trace(go.Bar(x=data.index, y=data["Volume"], name="Volume"))
    fig_volume.update_layout(title="Trading Volume Over Time", xaxis_title="Date", yaxis_title="Volume")

    fig_returns = go.Figure()
    fig_returns.add_trace(go.Scatter(x=data.index, y=data["Returns"], mode="lines", name="Daily Returns"))
    fig_returns.update_layout(title="Daily Returns Over Time", xaxis_title="Date", yaxis_title="Returns")

    fig_volatility = go.Figure()
    fig_volatility.add_trace(go.Scatter(x=data.index, y=data["Volatility"], mode="lines", name="Volatility"))
    fig_volatility.update_layout(title="20-Day Volatility Over Time", xaxis_title="Date", yaxis_title="Volatility")

    return fig_close, fig_volume, fig_returns, fig_volatility


# ─────────────────────────────────────────────────────────────────────────────
# ANOMALY DETECTION FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def detect_zscore_anomalies(data, threshold=3):
    z_scores = np.abs((data["Close"] - data["Close"].mean()) / data["Close"].std())
    return z_scores > threshold


def detect_iforest_anomalies(data):
    scaler = StandardScaler()
    X = scaler.fit_transform(data[["Close", "Volume", "Returns", "Volatility"]])
    iso_forest = IsolationForest(contamination=0.01, random_state=42)
    return iso_forest.fit_predict(X) == -1


def detect_dbscan_anomalies(data):
    scaler = StandardScaler()
    X = scaler.fit_transform(data[["Close", "Volume", "Returns", "Volatility"]])
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    return dbscan.fit_predict(X) == -1


def detect_lstm_anomalies(data, sequence_length=20, threshold_percentile=95):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[["Close", "Volume", "Returns", "Volatility"]])

    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i : (i + sequence_length)])
        y.append(scaled_data[i + sequence_length])
    X, y = np.array(X), np.array(y)

    model = Sequential([
        LSTM(50, activation="relu", input_shape=(sequence_length, 4)),
        Dense(4),
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    model.fit(X, y, epochs=50, batch_size=32, verbose=0)

    predictions = model.predict(X, verbose=0)
    mse = np.mean(np.power(y - predictions, 2), axis=1)
    threshold = np.percentile(mse, threshold_percentile)

    anomalies = np.zeros(len(data))
    anomalies[sequence_length:] = mse > threshold
    return anomalies.astype(bool)


def detect_autoencoder_anomalies(data, threshold_percentile=95):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[["Close", "Volume", "Returns", "Volatility"]])

    input_dim    = scaled_data.shape[1]
    encoding_dim = 2

    input_layer = Input(shape=(input_dim,))
    enc = Dense(8, activation="relu")(input_layer)
    enc = Dense(4, activation="relu")(enc)
    enc = Dense(encoding_dim, activation="relu")(enc)
    dec = Dense(4, activation="relu")(enc)
    dec = Dense(8, activation="relu")(dec)
    dec = Dense(input_dim, activation="linear")(dec)

    autoencoder = Model(inputs=input_layer, outputs=dec)
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    autoencoder.fit(scaled_data, scaled_data, epochs=100, batch_size=32,
                    shuffle=True, verbose=0)

    predictions = autoencoder.predict(scaled_data, verbose=0)
    mse = np.mean(np.power(scaled_data - predictions, 2), axis=1)
    threshold = np.percentile(mse, threshold_percentile)
    return mse > threshold


def create_anomaly_plot(data, anomalies, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data["Close"],
                             mode="lines", name="Close Price"))
    fig.add_trace(go.Scatter(x=data.index[anomalies], y=data["Close"][anomalies],
                             mode="markers", name="Anomalies",
                             marker=dict(color="red", size=8)))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Price")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT APP
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Stock Anomaly Detection", layout="wide")
st.title("Stock Price Anomaly Detection")

# Sidebar
ticker     = st.sidebar.text_input("Enter stock ticker", "GME")
start_date = st.sidebar.date_input("Start date", pd.to_datetime("2020-01-01"))
end_date   = st.sidebar.date_input("End date",   pd.to_datetime("2023-12-31"))

data = load_data(ticker, str(start_date), str(end_date))

if data.empty:
    st.error("No data returned. Check the ticker symbol and date range.")
    st.stop()

# ── EDA ───────────────────────────────────────────────────────────────────────
st.header("Exploratory Data Analysis")
fig_close, fig_volume, fig_returns, fig_volatility = create_eda_plots(data)

st.subheader("Closing Price with 20-day MA and Bollinger Bands")
st.plotly_chart(fig_close, use_container_width=True)

st.subheader("Trading Volume Over Time")
st.plotly_chart(fig_volume, use_container_width=True)

st.subheader("Daily Returns Over Time")
st.plotly_chart(fig_returns, use_container_width=True)

st.subheader("20-Day Volatility Over Time")
st.plotly_chart(fig_volatility, use_container_width=True)

st.subheader("Basic Statistics")
st.write(data.describe())

st.subheader("Correlation Matrix")
corr_matrix = data[["Close", "Volume", "Returns", "Volatility"]].corr()
fig_corr = go.Figure(data=go.Heatmap(
    z=corr_matrix.values,
    x=corr_matrix.columns.tolist(),
    y=corr_matrix.index.tolist(),
    colorscale="Viridis",
))
fig_corr.update_layout(title="Correlation Matrix")
st.plotly_chart(fig_corr, use_container_width=True)

# ── ANOMALY DETECTION ────────────────────────────────────────────────────────
st.header("Anomaly Detection")

with st.spinner("Running Z-Score..."):
    zscore_anomalies = detect_zscore_anomalies(data)
with st.spinner("Running Isolation Forest..."):
    iforest_anomalies = detect_iforest_anomalies(data)
with st.spinner("Running DBSCAN..."):
    dbscan_anomalies = detect_dbscan_anomalies(data)
with st.spinner("Running LSTM (this takes ~1-2 min)..."):
    lstm_anomalies = detect_lstm_anomalies(data)
with st.spinner("Running Autoencoder (this takes ~1-2 min)..."):
    autoencoder_anomalies = detect_autoencoder_anomalies(data)

st.subheader("Z-Score Anomalies")
st.plotly_chart(create_anomaly_plot(data, zscore_anomalies, "Z-Score Anomalies"),
                use_container_width=True)

st.subheader("Isolation Forest Anomalies")
st.plotly_chart(create_anomaly_plot(data, iforest_anomalies, "Isolation Forest Anomalies"),
                use_container_width=True)

st.subheader("DBSCAN Anomalies")
st.plotly_chart(create_anomaly_plot(data, dbscan_anomalies, "DBSCAN Anomalies"),
                use_container_width=True)

st.subheader("LSTM Anomalies")
st.plotly_chart(create_anomaly_plot(data, lstm_anomalies, "LSTM Anomalies"),
                use_container_width=True)

st.subheader("Autoencoder Anomalies")
st.plotly_chart(create_anomaly_plot(data, autoencoder_anomalies, "Autoencoder Anomalies"),
                use_container_width=True)

# ── COMBINED OVERLAY ──────────────────────────────────────────────────────────
st.subheader("All Models Comparison")
fig_all = go.Figure()
fig_all.add_trace(go.Scatter(x=data.index, y=data["Close"], mode="lines", name="Close Price"))
fig_all.add_trace(go.Scatter(x=data.index[zscore_anomalies],    y=data["Close"][zscore_anomalies],
                             mode="markers", name="Z-Score",
                             marker=dict(color="red",    size=8, symbol="circle")))
fig_all.add_trace(go.Scatter(x=data.index[iforest_anomalies],   y=data["Close"][iforest_anomalies],
                             mode="markers", name="Isolation Forest",
                             marker=dict(color="green",  size=8, symbol="square")))
fig_all.add_trace(go.Scatter(x=data.index[dbscan_anomalies],    y=data["Close"][dbscan_anomalies],
                             mode="markers", name="DBSCAN",
                             marker=dict(color="blue",   size=8, symbol="diamond")))
fig_all.add_trace(go.Scatter(x=data.index[lstm_anomalies],      y=data["Close"][lstm_anomalies],
                             mode="markers", name="LSTM",
                             marker=dict(color="purple", size=8, symbol="cross")))
fig_all.add_trace(go.Scatter(x=data.index[autoencoder_anomalies], y=data["Close"][autoencoder_anomalies],
                             mode="markers", name="Autoencoder",
                             marker=dict(color="orange", size=8, symbol="star")))
fig_all.update_layout(title="All Models Comparison", xaxis_title="Date", yaxis_title="Price")
st.plotly_chart(fig_all, use_container_width=True)

# ── SUMMARY TABLE ─────────────────────────────────────────────────────────────
st.subheader("Summary Statistics")
summary = pd.DataFrame({
    "Model": ["Z-Score", "Isolation Forest", "DBSCAN", "LSTM", "Autoencoder"],
    "Anomalies Detected": [
        int(zscore_anomalies.sum()),
        int(iforest_anomalies.sum()),
        int(dbscan_anomalies.sum()),
        int(lstm_anomalies.sum()),
        int(autoencoder_anomalies.sum()),
    ],
})
st.table(summary)

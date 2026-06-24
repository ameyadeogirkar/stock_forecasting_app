import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error

# -------------------------
# PAGE CONFIG
# -------------------------

st.set_page_config(
    page_title="Stock Forecasting Platform",
    layout="wide"
)

st.title("📈 Professional Stock Forecasting Dashboard")

# -------------------------
# SIDEBAR
# -------------------------

st.sidebar.header("Settings")

ticker = st.sidebar.text_input(
    "Ticker Symbol",
    value="AAPL"
)

forecast_days = st.sidebar.slider(
    "Forecast Days",
    30,
    365,
    180
)

show_rsi = st.sidebar.checkbox(
    "Show RSI",
    True
)

show_macd = st.sidebar.checkbox(
    "Show MACD",
    True
)

show_bbands = st.sidebar.checkbox(
    "Show Bollinger Bands",
    True
)

# -------------------------
# DATA DOWNLOAD
# -------------------------

@st.cache_data
def load_data(ticker):

    data = yf.download(
        ticker,
        period="5y",
        auto_adjust=True,
        progress=False
    )

    return data

df = load_data(ticker)

if df.empty:
    st.error("Invalid ticker.")
    st.stop()

# -------------------------
# INDICATORS
# -------------------------

close = df["Close"].squeeze()

close = pd.to_numeric(close, errors="coerce")

close = close.dropna()

close = close.replace([np.inf, -np.inf], np.nan)

close = close.dropna()

# Moving Average
df["MA20"] = close.rolling(20).mean()

# RSI

delta = close.diff()

gain = delta.where(delta > 0, 0)

loss = -delta.where(delta < 0, 0)

avg_gain = gain.rolling(14).mean()

avg_loss = loss.rolling(14).mean()

rs = avg_gain / avg_loss

df["RSI"] = 100 - (100 / (1 + rs))

# MACD

ema12 = close.ewm(span=12).mean()

ema26 = close.ewm(span=26).mean()

df["MACD"] = ema12 - ema26

df["Signal"] = df["MACD"].ewm(span=9).mean()

# Bollinger Bands

rolling_mean = close.rolling(20).mean()

rolling_std = close.rolling(20).std()

df["Upper"] = rolling_mean + (rolling_std * 2)

df["Lower"] = rolling_mean - (rolling_std * 2)

# -------------------------
# KPI CARDS
# -------------------------

current_price = float(close.iloc[-1])

return_pct = (
    (close.iloc[-1] - close.iloc[-252])
    / close.iloc[-252]
) * 100

volatility = (
    close.pct_change().std()
    * np.sqrt(252)
    * 100
)

col1, col2, col3 = st.columns(3)

col1.metric(
    "Current Price",
    f"${current_price:.2f}"
)

col2.metric(
    "1 Year Return",
    f"{return_pct:.2f}%"
)

col3.metric(
    "Volatility",
    f"{volatility:.2f}%"
)

# -------------------------
# PRICE CHART
# -------------------------

st.subheader("Historical Price")

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=df.index,
        y=close,
        name="Close"
    )
)

fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["MA20"],
        name="MA20"
    )
)

if show_bbands:

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Upper"],
            name="Upper Band"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Lower"],
            name="Lower Band"
        )
    )

fig.update_layout(
    height=600
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# -------------------------
# AUTO ARIMA
# -------------------------

st.subheader("Training Auto ARIMA")

with st.spinner("Building Forecast..."):

    model = auto_arima(
        close,
        seasonal=False,
        suppress_warnings=True
    )

    forecast, confint = model.predict(
        n_periods=forecast_days,
        return_conf_int=True
    )

future_dates = pd.date_range(
    start=close.index[-1],
    periods=forecast_days + 1,
    freq="D"
)[1:]

forecast_df = pd.DataFrame(
    {
        "Date": future_dates,
        "Forecast": forecast,
        "Lower": confint[:, 0],
        "Upper": confint[:, 1]
    }
)

# -------------------------
# FORECAST CHART
# -------------------------

st.write(close.head())
st.write(close.tail())
st.write(type(close))
st.write(close.isna().sum())

st.subheader("Forecast")

fig2 = go.Figure()

fig2.add_trace(
    go.Scatter(
        x=df.index,
        y=close,
        name="Historical"
    )
)

fig2.add_trace(
    go.Scatter(
        x=forecast_df["Date"],
        y=forecast_df["Forecast"],
        name="Forecast"
    )
)

fig2.add_trace(
    go.Scatter(
        x=forecast_df["Date"],
        y=forecast_df["Upper"],
        name="Upper CI"
    )
)

fig2.add_trace(
    go.Scatter(
        x=forecast_df["Date"],
        y=forecast_df["Lower"],
        name="Lower CI"
    )
)

fig2.update_layout(
    height=600
)

st.plotly_chart(
    fig2,
    use_container_width=True
)

# -------------------------
# RSI
# -------------------------

if show_rsi:

    st.subheader("RSI")

    rsi_fig = go.Figure()

    rsi_fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["RSI"],
            name="RSI"
        )
    )

    st.plotly_chart(
        rsi_fig,
        use_container_width=True
    )

# -------------------------
# MACD
# -------------------------

if show_macd:

    st.subheader("MACD")

    macd_fig = go.Figure()

    macd_fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["MACD"],
            name="MACD"
        )
    )

    macd_fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Signal"],
            name="Signal"
        )
    )

    st.plotly_chart(
        macd_fig,
        use_container_width=True
    )

# -------------------------
# FORECAST TABLE
# -------------------------

st.subheader("Forecast Data")

st.dataframe(
    forecast_df.tail(30),
    use_container_width=True
)

# -------------------------
# CSV DOWNLOAD
# -------------------------

csv = forecast_df.to_csv(
    index=False
)

st.download_button(
    "⬇ Download Forecast CSV",
    csv,
    file_name=f"{ticker}_forecast.csv",
    mime="text/csv"
)

# -------------------------
# SIMPLE SIGNAL
# -------------------------

latest_rsi = df["RSI"].iloc[-1]

if latest_rsi < 30:
    st.success("🟢 BUY SIGNAL (RSI Oversold)")
elif latest_rsi > 70:
    st.error("🔴 SELL SIGNAL (RSI Overbought)")
else:
    st.info("🟡 HOLD SIGNAL")

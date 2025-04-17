import streamlit as st
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# ---- SETTINGS ---- #
default_shares = 1650
rotation_percent = 0.20
thresholds = [600_000, 750_000, 1_000_000]
bayesian_prior = 0.515  # from previous simulation
num_simulations = 1000
n_years = 22
initial_price = 311
volatility = 0.7
expected_return = 0.25

# ---- STREAMLIT SETUP ---- #
st.set_page_config(page_title="MSTR Rotation Tracker", layout="wide")
st.title("📈 MSTR Rotation Tracker")

st.markdown("""
This tool helps long-term MSTR holders make probabilistic, data-driven rotation decisions using Bayesian logic and on-chain market signals.

> ⚠️ **Why use this?**
> To avoid over-indexing on single market moves, and instead rotate strategically using tools like STH-SOPA, STH-MVRV-Z, and Futures Funding Rates.
""")

# ---- LIVE PRICE DATA ---- #
btc_ticker = yf.Ticker("BTC-USD")
mstr_ticker = yf.Ticker("MSTR")

try:
    btc_price = btc_ticker.history(period="1d")['Close'].iloc[-1]
    mstr_price_live = mstr_ticker.history(period="1d")['Close'].iloc[-1]
except IndexError:
    st.error("Failed to retrieve live market data. Please try again later.")
    st.stop()

st.markdown(f"#### 📊 Live BTC Price: **${btc_price:,.0f}**")
st.markdown(f"#### 📈 Live MSTR Price: **${mstr_price_live:,.2f}**")

# ---- USER INPUTS ---- #

st.markdown("### 📉 On-Chain Market Signals")
sth_mvrv_z = st.number_input("Current STH-MVRV-Z Score", value=1.00, step=0.1,
    help="Z-score above 6 historically signals a topping structure; <1 implies undervaluation.")
funding_rate = st.number_input("Current Futures Funding Rate (%)", value=2.00, step=0.01,
    help="High values (e.g., >0.1%) may indicate overheated markets or bullish bias.")
shares_held = st.number_input("Shares Held", value=default_shares, step=10)
current_age = st.number_input("Your Age", value=48, step=1)
selected_threshold = st.selectbox("Rotation Trigger Threshold ($)", thresholds)

st.markdown("### 🔁 Allocation Preferences (Total = 100%)")
msty_pct = st.slider("% to MSTY (High Yield)", 0, 100, 50)
strk_pct = st.slider("% to STRK (Preferred)", 0, 100 - msty_pct, 25)
strf_pct = 100 - msty_pct - strk_pct
st.text(f"% to STRF (Preferred): {strf_pct}%")

msty_yield = 0.20
strk_yield = 0.075
strf_yield = 0.075

# ---- MONTE CARLO SIMULATION ---- #
initial_value = mstr_price_live * shares_held
dt = 1
np.random.seed(42)
simulations = np.zeros((n_years + 1, num_simulations))
simulations[0] = initial_value

for t in range(1, n_years + 1):
    rand = np.random.normal(0, 1, num_simulations)
    simulations[t] = simulations[t - 1] * np.exp((expected_return - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * rand)

# ---- ROTATION & INCOME CALCULATION ---- #
rotation_value = initial_value * rotation_percent
msty_income = rotation_value * (msty_pct / 100) * msty_yield
strk_income = rotation_value * (strk_pct / 100) * strk_yield
strf_income = rotation_value * (strf_pct / 100) * strf_yield
total_income = msty_income + strk_income + strf_income

# ---- PLOTTING ---- #
years = np.arange(current_age, current_age + n_years + 1)
mean_projection = simulations.mean(axis=1)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(years, mean_projection, label="Mean Portfolio Value", linewidth=2)
ax.axvline(current_age + 7, linestyle='--', color='gray', label='Full Retirement')
ax.fill_between(years, mean_projection - mean_projection.std(axis=0), mean_projection + mean_projection.std(axis=0), color='blue', alpha=0.1)
ax.set_title("Projected Portfolio Value with Rotation & Income")
ax.set_xlabel("Age")
ax.set_ylabel("Portfolio Value ($)")
ax.legend()
ax.grid(True)
st.pyplot(fig)

st.markdown(f"### 💸 Estimated Annual Income After Rotation: **${total_income:,.0f}**")

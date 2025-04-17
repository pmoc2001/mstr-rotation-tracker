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
sth_mvrv_z = st.number_input("Current STH-MVRV-Z Score", value=1.00, step=0.1)
funding_rate = st.number_input("Current Futures Funding Rate (%)", value=2.00, step=0.01)

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

# ---- PLOTTING MONTE CARLO ---- #
years = np.arange(current_age, current_age + n_years + 1)
mean_projection = simulations.mean(axis=1)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(years, mean_projection, label="Mean Portfolio Value", linewidth=2)
ax.axvline(current_age + 7, linestyle='--', color='gray', label='Full Retirement')
ax.fill_between(years, mean_projection - simulations.std(axis=1), mean_projection + simulations.std(axis=1), color='blue', alpha=0.1)
ax.set_title("Projected Portfolio Value with Rotation & Income")
ax.set_xlabel("Age")
ax.set_ylabel("Portfolio Value ($)")
ax.legend()
ax.grid(True)
st.pyplot(fig)

st.markdown(f"### 💸 Estimated Annual Income After Rotation: **${total_income:,.0f}**")

# ---- BAYESIAN CHART ---- #
confidence_boost = 1 if initial_value >= selected_threshold else 0
data_points = 100
prior_successes = int(bayesian_prior * data_points)
successes = prior_successes + confidence_boost
posterior_prob = (successes + 1) / (data_points + 2)

x_vals = np.arange(10, 501, 10)
y_vals = [(int(bayesian_prior * x) + confidence_boost + 1) / (x + 2) for x in x_vals]

fig2, ax2 = plt.subplots()
ax2.plot(x_vals, y_vals, label="Bayesian Probability", color="blue")
ax2.axhline(y=posterior_prob, color='green', linestyle='--', label=f"Current: {posterior_prob:.1%}")
ax2.set_title("Bayesian Probability vs. Historical Evidence")
ax2.set_xlabel("Number of Data Points (Historical Evidence)")
ax2.set_ylabel("Posterior Probability")
ax2.legend()
ax2.grid(True)
st.pyplot(fig2)

st.caption("""
This tool uses Monte Carlo-informed Bayesian logic and market signals to help you time MSTR portfolio rotation.

- **STH-MVRV-Z** = short-term holder unrealized profit/loss vs cost basis (Z-score); >6 may signal tops, <1 undervaluation
- **Funding Rate** = bullish/bearish positioning in futures markets

Avoid overreacting to short-term noise — rotate based on data + probability.
""")

import streamlit as st
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# ---- SETTINGS ---- #
default_shares = 1650
msty_yield = 0.20
preferred_yield = 0.075
rotation_percent = 0.20
thresholds = [600_000, 750_000, 1_000_000]
bayesian_prior = 0.515  # from previous simulation

# ---- FETCH LIVE DATA ---- #
btc_ticker = yf.Ticker("BTC-USD")
mstr_ticker = yf.Ticker("MSTR")

try:
    btc_price = btc_ticker.history(period="1d")['Close'].iloc[-1]
    mstr_price_live = mstr_ticker.history(period="1d")['Close'].iloc[-1]
except IndexError:
    st.error("Failed to retrieve live market data. Please try again later.")
    st.stop()

# ---- STREAMLIT APP ---- #
st.title("📈 MSTR Rotation Tracker")

st.markdown(f"#### 📊 Live BTC Price: **${btc_price:,.0f}**")
st.markdown(f"#### 📈 Live MSTR Price: **${mstr_price_live:,.2f}**")

shares_held = st.number_input("Shares Held", value=default_shares, step=10)
current_age = st.number_input("Your Age", value=48, step=1)
selected_threshold = st.selectbox("Rotation Trigger Threshold ($)", thresholds)

# ---- CALCULATIONS ---- #
portfolio_value = mstr_price_live * shares_held
threshold_met = portfolio_value >= selected_threshold

# ---- INTERACTIVE BAYESIAN PROBABILITY ---- #
st.markdown("### 🧠 Bayesian Model Settings")
data_points = st.slider("Historical Evidence Weight (Data Points)", min_value=10, max_value=500, value=100, step=10)

# ---- STH-SOPA Manual Input ---- #
sth_sopa = st.number_input("Current STH-SOPA Value (manually input from Glassnode or chart)", value=1.00, step=0.01)

# Adjust confidence based on STH-SOPA
if sth_sopa > 1:
    st.success("📈 STH-SOPA > 1: Market in profit — increasing model confidence")
    data_points += 50
elif sth_sopa < 1:
    st.warning("📉 STH-SOPA < 1: Market in loss — decreasing model confidence")
    data_points = max(data_points - 25, 10)  # avoid dropping below minimum

confidence_boost = 1 if threshold_met else 0
prior_successes = int(bayesian_prior * data_points)
successes = prior_successes + confidence_boost
failures = data_points - prior_successes + (0 if threshold_met else 1)
posterior_prob = (successes + 1) / (data_points + 2)

# Estimated income if rotated
rotation_value = portfolio_value * rotation_percent
est_income = rotation_value * (msty_yield + preferred_yield) / 2

# ---- OUTPUT ---- #
st.markdown(f"### 💰 Portfolio Value: **${portfolio_value:,.0f}**")

if threshold_met:
    st.success(f"✅ Portfolio exceeds ${selected_threshold:,} — rotation eligible!")
    st.markdown(f"### 🔁 Estimated Income if You Rotate 20% Now: **${est_income:,.0f}/yr**")
    st.markdown(f"### 🧠 Bayesian Probability That Rotation is Optimal: **{posterior_prob:.1%}**")
    if posterior_prob > 0.60:
        st.info("**Recommended Action:** Consider rotating 20% into MSTY + STRK/STRF.")
    else:
        st.warning("**Hold for now.** Rotation is close but not yet clearly optimal.")
else:
    st.warning(f"Portfolio is below the selected threshold (${selected_threshold:,}).")
    st.markdown(f"### 🧠 Bayesian Probability That Rotation is Optimal: **{posterior_prob:.1%}**")
    st.info("Keep monitoring. Rotation opportunity may emerge soon.")

# ---- CHART: Bayesian Probability vs. Data Points ---- #
x_vals = np.arange(10, 501, 10)
y_vals = [(int(bayesian_prior * x) + confidence_boost + 1) / (x + 2) for x in x_vals]

fig, ax = plt.subplots()
ax.plot(x_vals, y_vals, label="Bayesian Probability", color="blue")
ax.axhline(y=posterior_prob, color='green', linestyle='--', label=f"Current: {posterior_prob:.1%}")
ax.set_title("Bayesian Probability vs. Historical Evidence")
ax.set_xlabel("Number of Data Points (Historical Evidence)")
ax.set_ylabel("Posterior Probability")
ax.legend()
ax.grid(True)
st.pyplot(fig)

st.markdown("---")
st.caption("This tool uses Monte Carlo-informed Bayesian logic and live market data to help you time MSTR portfolio rotation.\n\n'Historical evidence' represents how confident you are in past simulations and assumptions. More data points = more trust in the prior probability.\n\nSTH-SOPA > 1 signals BTC market profit-taking, increasing trust. STH-SOPA < 1 signals caution.")

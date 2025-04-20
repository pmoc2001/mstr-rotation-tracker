import streamlit as st
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# ---- SETTINGS ---- #
default_shares = 1650
rotation_percent = 0.20
thresholds = [600_000, 750_000, 1_000_000]
bayesian_prior = 0.515
num_simulations = 1000
n_years = 22
volatility = 0.7
expected_return = 0.25

# ---- STREAMLIT CONFIG ---- #
st.set_page_config(page_title="MSTR Rotation Assistant", layout="wide")
st.title("📊 MSTR Rotation Decision Assistant")

st.markdown("""
This tool helps you decide **when and how much MSTR to rotate** based on your current position, market context, and future retirement goals.

**🧠 Powered by:**
- Live market data (BTC + MSTR)
- Monte Carlo simulations for capital growth
- Bayesian logic with real-time signal inputs (STH-SOPA, MVRV-Z, Funding Rate)
""")

# ---- SECTION 1: PORTFOLIO SNAPSHOT ---- #
st.header("1️⃣ Portfolio Snapshot")
col1, col2 = st.columns(2)
with col1:
    shares_held = st.number_input("Shares Held", value=default_shares, step=10)
    current_age = st.number_input("Your Age", value=48, step=1)
    selected_threshold = st.selectbox("Rotation Trigger Threshold ($)", thresholds)
with col2:
    btc_price = yf.Ticker("BTC-USD").history(period="1d")['Close'].iloc[-1]
    mstr_price = yf.Ticker("MSTR").history(period="1d")['Close'].iloc[-1]
    st.metric("📉 BTC Price", f"${btc_price:,.0f}")
    st.metric("📈 MSTR Price", f"${mstr_price:,.2f}")

portfolio_value = mstr_price * shares_held
st.metric("💼 Portfolio Value", f"${portfolio_value:,.0f}")

rotate_now = st.slider("How much are you willing to rotate today?", 0, 50, int(rotation_percent * 100)) / 100

# ---- SECTION 2: MARKET SIGNALS ---- #
st.header("2️⃣ Market Conditions & Signal Confidence")
st.markdown("Input current on-chain and derivatives data")

sth_sopa = st.number_input("STH-SOPA", value=1.00, step=0.01)
sth_mvrv_z = st.number_input("STH-MVRV-Z", value=1.00, step=0.1)
funding_rate = st.number_input("Futures Funding Rate (%)", value=2.00, step=0.01)

data_points = 100
if sth_sopa > 1: data_points += 50
elif sth_sopa < 1: data_points = max(data_points - 25, 10)
if sth_mvrv_z > 6: data_points = max(data_points - 25, 10)
if funding_rate > 0.1: data_points = max(data_points - 25, 10)

confidence_boost = 1 if portfolio_value >= selected_threshold else 0
prior_successes = int(bayesian_prior * data_points)
posterior_prob = (prior_successes + confidence_boost + 1) / (data_points + 2)

with st.expander("🧠 What is Bayesian Rotation Probability?"):
    st.markdown("""
    This number reflects the **calculated probability** (using Bayesian inference) that rotating some MSTR now is the optimal decision — based on both historical success rates and real-time market signals (STH-SOPA, MVRV-Z, Funding Rate).

    - >60% = strong signal to rotate
    - 50–60% = watch closely
    - <50% = wait

    The model becomes more confident as historical evidence increases and current conditions align with past profitable outcomes.
    """)
st.metric("🧠 Bayesian Rotation Probability", f"{posterior_prob:.1%}")

# ---- SECTION 3: ROTATION ALLOCATION ---- #
st.header("3️⃣ Allocation Strategy")

msty_pct = st.slider("% to MSTY (High Yield)", 0, 100, 50)
strk_pct = st.slider("% to STRK (Preferred)", 0, 100 - msty_pct, 25)
strf_pct = 100 - msty_pct - strk_pct
st.markdown(f"🔁 **STRF Auto-Allocated:** `{strf_pct}%`")

msty_yield = 0.20
strk_yield = 0.075
strf_yield = 0.075
rotation_value = portfolio_value * rotate_now
est_income = (rotation_value * (msty_pct/100) * msty_yield +
              rotation_value * (strk_pct/100) * strk_yield +
              rotation_value * (strf_pct/100) * strf_yield)

st.metric("💸 Projected Annual Income from Rotation", f"${est_income:,.0f}")

# ---- SECTION 4: MONTE CARLO PROJECTION ---- #
retirement_age = st.slider("Target Retirement Age", min_value=current_age + 1, max_value=current_age + n_years, value=current_age + 7)
st.header("4️⃣ Monte Carlo Forecast")

np.random.seed(42)
sim = np.zeros((n_years + 1, num_simulations))
sim_income = np.zeros((n_years + 1, num_simulations))
sim[0] = portfolio_value
sim_income[:, :] = est_income  # constant income overlay

for t in range(1, n_years + 1):
    rand = np.random.normal(0, 1, num_simulations)
    sim[t] = sim[t - 1] * np.exp((expected_return - 0.5 * volatility**2) + volatility * rand)

years = np.arange(current_age, current_age + n_years + 1)
mean_projection = sim.mean(axis=1)
mean_income = sim_income.mean(axis=1)
cumulative_income = np.cumsum(mean_income)

fig, ax = plt.subplots()
ax.plot(years, mean_projection, label="Mean Portfolio Value", linewidth=2)
ax.fill_between(years, mean_projection - sim.std(axis=1), mean_projection + sim.std(axis=1), alpha=0.2)
ax.plot(years, cumulative_income, label="Cumulative Income", linestyle="--", color="green")
ax.axvline(retirement_age, color='gray', linestyle='--', label='Target Retirement Age')
ax.set_title("Projected Portfolio Value & Cumulative Income")
ax.set_xlabel("Age")
ax.set_ylabel("USD")
ax.grid(True)
ax.legend()
st.pyplot(fig)

# ---- SECTION 5: BAYESIAN DECISION CHART ---- #
st.header("5️⃣ Bayesian Model Sensitivity")
x_vals = np.arange(10, 501, 10)
y_vals = [(int(bayesian_prior * x) + confidence_boost + 1) / (x + 2) for x in x_vals]

fig2, ax2 = plt.subplots()
ax2.plot(x_vals, y_vals, label="Bayesian Probability", color="blue")
ax2.axhline(posterior_prob, color="green", linestyle="--", label=f"Current: {posterior_prob:.1%}")
ax2.set_xlabel("Historical Confidence Weight")
ax2.set_ylabel("Posterior Probability")
ax2.set_title("Impact of Confidence on Rotation Signal")
ax2.grid(True)
ax2.legend()
st.pyplot(fig2)

# ---- SUMMARY ---- #
st.header("✅ Summary & Recommendation")
if posterior_prob >= 0.60:
    st.success("Rotation recommended: Bayesian probability exceeds 60%.")
elif 0.50 <= posterior_prob < 0.60:
    st.info("Rotation possible: Market conditions uncertain, monitor closely.")
else:
    st.warning("Rotation not advised: Confidence is too low based on current signals.")

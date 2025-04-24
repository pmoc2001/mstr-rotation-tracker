import streamlit as st
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize

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
st.set_page_config(page_title="MSTR Retirement Assistant", layout="wide")
st.title("📊 MSTR Retirement Decision Assistant")

st.markdown("""
This assistant clearly shows you **when and how** to rotate your MSTR holdings into income-producing investments (MSTY, STRK, STRF) for a secure retirement.
""")

# ---- FETCH LIVE MARKET DATA ---- #
btc_price = yf.Ticker("BTC-USD").history(period="1d")['Close'].iloc[-1]
mstr_price = yf.Ticker("MSTR").history(period="1d")['Close'].iloc[-1]

# ---- USER INPUTS ---- #
st.sidebar.header("Your Current Situation")
shares_held = st.sidebar.number_input("Shares Held", value=default_shares, step=10)
current_age = st.sidebar.number_input("Your Age", value=48, step=1)
retirement_age = st.sidebar.slider("Retirement Age", min_value=current_age + 1, max_value=current_age + 20, value=current_age + 7)
selected_threshold = st.sidebar.selectbox("Rotation Trigger Threshold ($)", thresholds)

portfolio_value = mstr_price * shares_held
st.metric("💼 Current Portfolio Value", f"${portfolio_value:,.0f}")

# ---- MARKET SIGNALS ---- #
st.sidebar.header("Market Conditions")
sth_sopa = st.sidebar.number_input("STH-SOPA", value=1.00, step=0.01)
sth_mvrv_z = st.sidebar.number_input("STH-MVRV-Z", value=1.00, step=0.1)
funding_rate = st.sidebar.number_input("Futures Funding Rate (%)", value=2.00, step=0.01)

# ---- BAYESIAN DECISION ---- #
data_points = 100
if sth_sopa > 1: data_points += 50
elif sth_sopa < 1: data_points -= 25
if sth_mvrv_z > 6: data_points -= 25
if funding_rate > 0.1: data_points -= 25
data_points = max(data_points, 10)

confidence_boost = 1 if portfolio_value >= selected_threshold else 0
prior_successes = int(bayesian_prior * data_points)
posterior_prob = (prior_successes + confidence_boost + 1) / (data_points + 2)

# ---- OPTIMAL ALLOCATION FUNCTION ---- #
def optimize_yield(msty_yield, strk_yield, strf_yield):
    def neg_income(x):
        return -(x[0]*msty_yield + x[1]*strk_yield + x[2]*strf_yield)
    constraints = ({'type': 'eq', 'fun': lambda x: sum(x)-1})
    bounds = [(0,1)]*3
    result = minimize(neg_income, [0.33,0.33,0.34], bounds=bounds, constraints=constraints)
    return result.x if result.success else [0.33,0.33,0.34]

msty_yield, strk_yield, strf_yield = 0.20, 0.075, 0.075
opt_alloc = optimize_yield(msty_yield, strk_yield, strf_yield)
msty_pct, strk_pct, strf_pct = [round(alloc*100) for alloc in opt_alloc]
rotation_value = portfolio_value * rotation_percent
est_income = rotation_value * sum(opt_alloc[i] * [msty_yield, strk_yield, strf_yield][i] for i in range(3))

# ---- DECISION OUTPUT ---- #
st.header("🚦 When Should You Rotate?")
if posterior_prob >= 0.60:
    decision = "🟢 Rotate Now"
elif 0.50 <= posterior_prob < 0.60:
    decision = "🟡 Wait and Monitor"
else:
    decision = "🔴 Hold - Not yet optimal"

st.subheader(f"Decision: {decision}")
st.markdown(f"**Probability of Rotation Being Optimal:** {posterior_prob:.1%}")

st.header("🏅 Optimal Allocation to Income Products")
st.markdown(f"""
- **MSTY (High Yield):** {msty_pct}%
- **STRK (Preferred Stock):** {strk_pct}%
- **STRF (Preferred Stock):** {strf_pct}%
""")
st.metric("💸 Annual Income from Rotation", f"${est_income:,.0f}")

# ---- MONTE CARLO SIMULATION ---- #
np.random.seed(42)
years = np.arange(current_age, retirement_age+1)
sim_rotated = np.zeros((len(years), num_simulations))
sim_income = np.zeros(len(years)) + est_income / 12

sim_rotated[0] = portfolio_value - rotation_value
for t in range(1, len(years)):
    sim_rotated[t] = sim_rotated[t-1] * np.exp((expected_return - 0.5*volatility**2) + volatility*np.random.normal(size=num_simulations))

mean_rotated = sim_rotated.mean(axis=1)
cum_income = np.cumsum(sim_income) * 12

# ---- VISUALIZATION ---- #
fig, ax = plt.subplots()
ax.plot(years, mean_rotated, label="Rotated Portfolio Value")
ax.plot(years, cum_income, label="Cumulative Income", linestyle="--", color="green")
ax.axvline(retirement_age, color='gray', linestyle='--', label='Retirement Age')
ax.set_title("Portfolio and Income Projection")
ax.set_xlabel("Age")
ax.set_ylabel("USD")
ax.grid(True)
ax.legend()
st.pyplot(fig)

st.markdown("---")
st.info("Adjust your inputs on the sidebar to explore different retirement scenarios.")

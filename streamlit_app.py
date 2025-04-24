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
msty_yield, strk_yield, strf_yield = 0.20, 0.075, 0.075

# ---- STREAMLIT CONFIG ---- #
st.set_page_config(page_title="MSTR Retirement Assistant", layout="wide")
st.title("📊 MSTR Retirement Decision Assistant")

st.markdown("""
This assistant helps you decide **when and how** to rotate your MSTR holdings into income-generating investments (MSTY, STRK, STRF) clearly and intuitively.
""")

# ---- FETCH LIVE MARKET DATA ---- #
btc_price = yf.Ticker("BTC-USD").history(period="1d")['Close'].iloc[-1]
mstr_price = yf.Ticker("MSTR").history(period="1d")['Close'].iloc[-1]

# ---- USER INPUTS ---- #
st.sidebar.header("Your Current Situation")
shares_held = st.sidebar.number_input("Shares Held", value=default_shares, step=10)
current_age = st.sidebar.number_input("Your Age", value=48, step=1)
retirement_age = st.sidebar.slider("Retirement Age", min_value=current_age + 1, max_value=current_age + 20, value=current_age + 7)
selected_threshold = st.sidebar.selectbox("Rotation Threshold ($)", thresholds)

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

# ---- OPTIMIZATION FUNCTION ---- #
def optimize_yield():
    def neg_income(x):
        return -(x[0]*msty_yield + x[1]*strk_yield + x[2]*strf_yield)
    constraints = ({'type': 'eq', 'fun': lambda x: sum(x)-1})
    bounds = [(0,1)]*3
    result = minimize(neg_income, [0.33,0.33,0.34], bounds=bounds, constraints=constraints)
    return result.x if result.success else [0.33,0.33,0.34]

optimal_alloc = optimize_yield()
msty_pct, strk_pct, strf_pct = [round(alloc*100) for alloc in optimal_alloc]

# ---- ALLOCATION OVERRIDE ---- #
st.header("🔀 Allocation to Income Products")
if st.checkbox("Manually Adjust Allocation"):

    msty_pct = st.slider("MSTY (%)", 0, 100, msty_pct)

    # Dynamically calculate slider max for STRK based on MSTY
    max_strk_pct = 100 - msty_pct
    strk_default = min(strk_pct, max_strk_pct)
    strk_pct = st.slider("STRK (%)", 0, max_strk_pct, strk_default)

    # Automatically assign remainder to STRF
    strf_pct = 100 - msty_pct - strk_pct
    st.write(f"STRF (%): {strf_pct}%")

    # Safety check in case of rounding issues
    if strf_pct < 0:
        st.error("Adjust sliders: Total allocation cannot exceed 100%.")
        strf_pct = 0

rotation_value = portfolio_value * rotation_percent
est_income = rotation_value * (msty_pct/100*msty_yield + strk_pct/100*strk_yield + strf_pct/100*strf_yield)

st.metric("💸 Annual Income from Rotation", f"${est_income:,.0f}")

# ---- DECISION TIMELINE ---- #
st.header("📅 Rotation Timeline")
fig, ax = plt.subplots(figsize=(10, 2))
ax.axvline(current_age, color='blue', linestyle='-', label='Today')
ax.axvline(retirement_age, color='gray', linestyle='--', label='Retirement')
if posterior_prob >= 0.60:
    decision_label, decision_color = 'Rotate Now', 'green'
elif posterior_prob >= 0.50:
    decision_label, decision_color = 'Wait & Monitor', 'orange'
else:
    decision_label, decision_color = 'Hold', 'red'
ax.text((current_age + retirement_age) / 2, 0.5, decision_label, fontsize=12, color=decision_color, ha='center')
ax.set_xlim(current_age - 1, retirement_age + 1)
ax.get_yaxis().set_visible(False)
ax.set_xlabel("Age")
ax.legend()
st.pyplot(fig)

# ---- MONTE CARLO SIMULATION ---- #
np.random.seed(42)
years = np.arange(current_age, retirement_age + 1)
sim_rotated = np.zeros((len(years), num_simulations))
sim_income = np.full(len(years), est_income)

sim_rotated[0] = portfolio_value - rotation_value
for t in range(1, len(years)):
    sim_rotated[t] = sim_rotated[t-1] * np.exp((expected_return - 0.5*volatility**2) + volatility*np.random.normal(size=num_simulations))

mean_rotated = sim_rotated.mean(axis=1)

# ---- VISUALIZATION ---- #
fig2, ax2 = plt.subplots()
ax2.plot(years, mean_rotated, label="Portfolio Value")
bars = ax2.bar(years, sim_income, alpha=0.5, label="Annual Income")
for bar in bars:
    height = bar.get_height()
    ax2.annotate(f'${height:,.0f}', xy=(bar.get_x() + bar.get_width()/2, height), xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=8)
ax2.axvline(retirement_age, linestyle='--', color='gray', label='Retirement Age')
ax2.set_title("Portfolio Value & Annual Income")
ax2.set_xlabel("Age")
ax2.set_ylabel("USD")
ax2.legend()
ax2.ticklabel_format(style='plain', axis='y')
st.pyplot(fig2)

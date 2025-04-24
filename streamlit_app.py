import streamlit as st
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ---- SETTINGS ---- #
default_shares = 1650
rotation_percent = 0.20               # <— Define this upfront
bayesian_prior = 0.515
num_simulations = 1000
volatility = 0.7
expected_return = 0.25
msty_yield, strk_yield, strf_yield = 0.20, 0.075, 0.075

# ---- STREAMLIT CONFIG ---- #
st.set_page_config(page_title="MSTR Retirement Assistant", layout="wide")
st.title("📊 MSTR Retirement Decision Assistant")
st.markdown("""
This assistant tells you **when** to rotate a portion of your MSTR into income assets, and **how much** to allocate to MSTY, STRK, and STRF—balancing income vs. growth.
""")

# ---- LIVE MARKET DATA ---- #
btc_price = yf.Ticker("BTC-USD").history(period="1d")['Close'].iloc[-1]
mstr_price = yf.Ticker("MSTR").history(period="1d")['Close'].iloc[-1]

# ---- USER INPUTS ---- #
st.sidebar.header("Your Situation")
shares = st.sidebar.number_input("Shares Held", value=default_shares, step=10)
age = st.sidebar.number_input("Current Age", value=48, min_value=18, max_value=80, step=1)
retire_age = st.sidebar.slider("Retirement Age", min_value=age+1, max_value=age+30, value=age+7)
threshold = st.sidebar.selectbox("Rotation Threshold ($)", [600_000, 750_000, 1_000_000])

st.sidebar.header("Market Signals")
sth_sopa    = st.sidebar.number_input("STH-SOPA", value=1.00, step=0.01)
sth_mvrv_z  = st.sidebar.number_input("STH-MVRV-Z", value=1.00, step=0.1)
fund_rate   = st.sidebar.number_input("Futures Funding Rate (%)", value=2.00, step=0.01)
inc_pref    = st.sidebar.slider("Income Preference (%)", 0, 100, 50)

# ---- PORTFOLIO VALUE ---- #
current_value = mstr_price * shares
st.metric("💼 Portfolio Value", f"${current_value:,.0f}")

# ---- BAYESIAN PROBABILITY ---- #
data_pts = 100
data_pts +=  (50 if sth_sopa   > 1 else -25 if sth_sopa   < 1 else 0)
data_pts += (-25 if sth_mvrv_z > 6 else 0)
data_pts += (-25 if fund_rate  > 0.1 else 0)
data_pts = max(data_pts, 10)

age_frac = np.clip(1 - (retire_age-age)/30, 0, 1)
boost = int(age_frac * 2)
if current_value >= threshold:
    boost += 1

prior_successes = int(bayesian_prior * data_pts)
posterior = (prior_successes + boost + 1) / (data_pts + 2)

# ---- DETERMINE ROTATION AGE ---- #
if posterior >= 0.60:
    action, rot_age = "Rotate Now", age
elif posterior >= 0.50:
    # find first age where projected value hits threshold
    for future in range(age+1, retire_age+1):
        proj_val = current_value * np.exp(expected_return*(future-age))
        if proj_val >= threshold:
            rot_age = future
            break
    else:
        rot_age = retire_age
    action = f"Rotate at age {rot_age}"
else:
    action, rot_age = "Hold Until Retirement", retire_age

# ---- ALLOCATION OPTIMIZER ---- #
def objective(x):
    # project to rotation age
    proj = current_value * np.exp(expected_return*(rot_age-age))
    # capital left after rotation
    cap_retire = proj * (1 - rotation_percent)
    # annual income
    ann_inc = proj*rotation_percent*(x[0]*msty_yield + x[1]*strk_yield + x[2]*strf_yield)
    years_ret = max(0, 90 - retire_age)
    # blended score
    score = (inc_pref/100)*ann_inc*years_ret + ((100-inc_pref)/100)*cap_retire
    return -score

bounds = [(0,1)]*3
cons   = ({'type':'eq','fun':lambda x: sum(x)-1},)
res    = minimize(objective, [1/3,1/3,1/3], bounds=bounds, constraints=cons)
opt    = res.x if res.success else [1/3,1/3,1/3]
msty_pct, strk_pct, strf_pct = [int(100*v) for v in opt]

# ---- MANUAL OVERRIDE ---- #
st.header("🔀 Allocation to Income Assets")
if st.checkbox("Manual Allocation"):
    msty_pct = st.slider("MSTY (%)", 0, 100, msty_pct)
    max_strk = 100 - msty_pct
    if max_strk > 0:
        strk_pct = st.slider("STRK (%)", 0, max_strk, strk_pct if strk_pct<=max_strk else max_strk)
    else:
        strk_pct = 0
    strf_pct = 100 - msty_pct - strk_pct
    if strf_pct < 0:
        st.error("Sum exceeds 100%, adjust sliders.")
        strf_pct = 0
else:
    st.write(f"Optimized Allocation: MSTY {msty_pct}%, STRK {strk_pct}%, STRF {strf_pct}%")

st.progress((msty_pct+strk_pct+strf_pct)/100)

# ---- ROTATION & INCOME ---- #
years_to_rot = rot_age - age
# For simplicity, use expected_return growth to project value
proj_value = current_value * np.exp(expected_return * years_to_rot)
rotation_amt = proj_value * rotation_percent
ann_income   = rotation_amt * (msty_pct/100*msty_yield + strk_pct/100*strk_yield + strf_pct/100*strf_yield)

st.metric("🚀 Action", action)
st.metric("📅 Rotation Age", rot_age)
st.metric("💸 Annual Income", f"${ann_income:,.0f}")

# ---- TIMELINE ---- #
st.header("📅 Timeline")
fig, ax = plt.subplots(figsize=(10,2))
ax.axvline(age

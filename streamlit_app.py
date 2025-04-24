import streamlit as st
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ---- SETTINGS ---- #
default_shares = 1650
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

# market signals
st.sidebar.header("Market Signals")
sth_sopa = st.sidebar.number_input("STH-SOPA", value=1.00, step=0.01)
sth_mvrv_z = st.sidebar.number_input("STH-MVRV-Z", value=1.00, step=0.1)
fund_rate = st.sidebar.number_input("Futures Funding Rate (%)", value=2.00, step=0.01)

# compute current portfolio
current_value = mstr_price * shares
st.metric("💼 Portfolio Value", f"${current_value:,.0f}")

# ---- BAYESIAN PROBABILITY ---- #
# base data points
data_pts = 100
data_pts += 50 if sth_sopa > 1 else -25 if sth_sopa < 1 else 0
data_pts += -25 if sth_mvrv_z > 6 else 0
data_pts += -25 if fund_rate > 0.1 else 0
data_pts = max(data_pts, 10)

# age-based boost
age_frac = np.clip(1 - (retire_age-age)/30, 0, 1)
boost = int(age_frac*2)
if current_value >= threshold: boost += 1

prior_successes = int(bayesian_prior * data_pts)
posterior = (prior_successes + boost + 1) / (data_pts + 2)

# ---- DETERMINE ROTATION AGE ---- #
if posterior >= 0.60:
    action, rot_age = "Rotate Now", age
elif posterior >= 0.50:
    # find first future age crossing 0.6
    for future in range(age+1, retire_age):
        # simplistic: assume same growth and signals
        # in practice you’d recalc posterior each year
        if current_value * np.exp(expected_return*(future-age)) >= threshold:
            rot_age = future
            break
    else:
        rot_age = retire_age
    action = f"Rotate at age {rot_age}"
else:
    action, rot_age = "Hold Until Retirement", retire_age

# ---- ALLOCATION OPTIMIZER ---- #
# slider for income vs growth preference
inc_pref = st.sidebar.slider("Income Preference (%)", 0, 100, 50)

def objective(x):
    # simulate projected capital at retirement
    proj = current_value * np.exp(expected_return*(rot_age-age))
    cap_retire = (proj*(1-rotation_percent))
    # income over retirement years
    years_ret = max(0, 90-retire_age)
    income_ann = proj*rotation_percent*(x[0]*msty_yield + x[1]*strk_yield + x[2]*strf_yield)
    score = (inc_pref/100)*income_ann*years_ret + ((100-inc_pref)/100)*cap_retire
    return -score

bounds = [(0,1)]*3
cons = ({'type':'eq','fun':lambda x: sum(x)-1},)
res = minimize(objective, [1/3,1/3,1/3], bounds=bounds, constraints=cons)
opt = res.x if res.success else [1/3,1/3,1/3]
msty_pct, strk_pct, strf_pct = [int(100*v) for v in opt]

# ---- ALLOCATION OVERRIDE ---- #
st.header("🔀 Allocation to Income Assets")
if st.checkbox("Manual Allocation"):
    msty_pct = st.slider("MSTY (%)", 0, 100, msty_pct)
    max_strk = 100-msty_pct
    if max_strk>0:
        strk_pct = st.slider("STRK (%)", 0, max_strk, min(strk_pct, max_strk))
    else:
        strk_pct = 0
    strf_pct = 100-msty_pct-strk_pct
    if strf_pct<0:
        st.error("Total >100%, adjust sliders.")
        strf_pct=0
else:
    st.write(f"Optimized: MSTY {msty_pct}%, STRK {strk_pct}%, STRF {strf_pct}%")

# show sum bar
st.progress((msty_pct+strk_pct+strf_pct)/100)

# ---- INCOME & ROTATION VALUE ---- #
rotation_percent = 0.20
# project growth to rotation age
years_to_rot = rot_age - age
proj_value = current_value * np.exp((expected_return-0.5*volatility**2)*years_to_rot
               + volatility*np.sqrt(years_to_rot)*np.random.normal())
rotation_amt = proj_value * rotation_percent

ann_income = rotation_amt*(msty_pct/100*msty_yield + strk_pct/100*strk_yield + strf_pct/100*strf_yield)
st.metric("🚀 Action", action)
st.metric("📅 Rotation Age", rot_age)
st.metric("💸 Annual Income", f"${ann_income:,.0f}")

# ---- DECISION TIMELINE ---- #
st.header("📅 Timeline")
fig,ax = plt.subplots(figsize=(10,2))
ax.axvline(age, color='blue', label="Today")
ax.axvline(rot_age, color='green', label="Rotation")
ax.axvline(retire_age, color='gray', linestyle='--', label="Retire")
ax.text((age+rot_age)/2,0.5,action,ha='center',color='black')
ax.set_xlim(age-1, retire_age+1); ax.get_yaxis().set_visible(False)
ax.legend(); st.pyplot(fig)

# ---- MONTE CARLO: HOLD→ROTATE→RETIRE ---- #
years1 = rot_age-age+1
years2 = retire_age-rot_age+1
sim = np.zeros((years1+years2, num_simulations))
# phase1: hold
sim[0] = current_value
for t in range(1,years1):
    sim[t] = sim[t-1]*np.exp((expected_return-0.5*volatility**2)+volatility*np.random.randn(num_simulations))
# apply rotation
sim[years1:] = (sim[years1-1]*(1-rotation_percent))
# phase2: post-rotation growth + income accumulation
income_ann = ann_income
for t in range(years1, years1+years2):
    sim[t] = sim[t-1]*np.exp((expected_return-0.5*volatility**2)+volatility*np.random.randn(num_simulations))
    
mean_path = sim.mean(axis=1)
ages = np.arange(age, retire_age+1)

# ---- VISUALIZATION ---- #
st.header("📈 Outlook")
fig2, ax2 = plt.subplots()
ax2.plot(ages, mean_path, label="Capital")
ax2.bar(ages, [income_ann if i>=years1 else 0 for i in range(len(ages))],
        alpha=0.4, label="Income")
for i,val in enumerate([income_ann if i>=years1 else 0 for i in range(len(ages))]):
    if val>0:
        ax2.text(ages[i], val, f"${val:,.0f}", ha='center', va='bottom', fontsize=8)
ax2.axvline(retire_age, color='gray', linestyle='--')
ax2.set_xlabel("Age"); ax2.set_ylabel("USD"); ax2.legend(); ax2.ticklabel_format(style='plain',axis='y')
st.pyplot(fig2)

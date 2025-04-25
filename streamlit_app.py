import streamlit as st
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ---- SETTINGS ---- #
default_shares   = 1650
rotation_percent = 0.20
bayesian_prior   = 0.515
num_simulations  = 500
volatility       = 0.7
msty_yield       = 0.20
strk_yield       = 0.075
strf_yield       = 0.075

# ---- PAGE CONFIG ---- #
st.set_page_config(page_title="MSTR Retirement Assistant", layout="wide")
st.title("📊 MSTR Retirement Decision Assistant")
st.markdown("Decide **when** to rotate part of your MSTR into income assets, and **how much** to allocate—balancing income vs. risk.")

# ---- LIVE PRICES ---- #
btc_price  = yf.Ticker("BTC-USD").history(period="1d")['Close'].iloc[-1]
mstr_price = yf.Ticker("MSTR").history(period="1d")['Close'].iloc[-1]

# ---- SIDEBAR INPUTS ---- #
with st.sidebar:
    st.header("Your Profile")
    shares     = st.number_input("Shares Held", value=default_shares, step=10)
    age        = st.number_input("Current Age", value=48, min_value=18, max_value=80, step=1)
    retire_age = st.slider("Retirement Age", age+1, age+30, age+7)
    threshold  = st.selectbox("Rotation Threshold ($)", [600_000, 750_000, 1_000_000])

    st.header("Market Signals")
    sth_sopa   = st.number_input("STH-SOPA", 1.00, step=0.01)
    sth_mvrv_z = st.number_input("STH-MVRV-Z", 1.00, step=0.1)
    fund_rate  = st.number_input("Futures Funding Rate (%)", 2.00, step=0.01)

    st.header("Preferences & Assumptions")
    inc_pref   = st.slider("Income Preference (%)", 0, 100, 50)
    btc_return = st.slider("BTC Expected Annual Return (%)", 0.0, 50.0, 15.0, 0.1) / 100
    risk_aversion = st.slider("Risk Aversion", 0.0, 1e-5, 1e-6, 1e-7)

# ---- PORTFOLIO VALUE ---- #
current_value = mstr_price * shares
st.metric("💼 Portfolio Value", f"${current_value:,.0f}")

# ---- BAYESIAN PROBABILITY ---- #
data_pts = 100
data_pts += 50 if sth_sopa > 1 else -25 if sth_sopa < 1 else 0
data_pts += -25 if sth_mvrv_z > 6 else 0
data_pts += -25 if fund_rate > 0.1 else 0
data_pts = max(data_pts, 10)

age_frac = np.clip(1 - (retire_age - age) / 30, 0, 1)
boost    = int(age_frac * 2) + (1 if current_value >= threshold else 0)
prior_succ = int(bayesian_prior * data_pts)
posterior  = (prior_succ + boost + 1) / (data_pts + 2)

# ---- ROTATION DECISION ---- #
if posterior >= 0.60:
    action, rot_age = "Rotate Now", age
elif posterior >= 0.50:
    action, rot_age = "Rotate Later", None
else:
    action, rot_age = "Hold Until Retirement", retire_age

st.subheader("🔁 Decision")
color = "green" if posterior >= 0.6 else "orange" if posterior >= 0.5 else "red"
st.markdown(f"**Rotation Probability:** <span style='color:{color}'>**{posterior:.1%}**</span>", unsafe_allow_html=True)
st.markdown(f"**Action:** **{action}**")
if rot_age is not None:
    st.markdown(f"**Rotation Age:** **{rot_age}**")

# ---- ALLOCATION OPTIMIZER (RISK-ADJUSTED) ---- #
def score_alloc(x):
    # Determine rotation age for projection
    target_age = age if action == "Rotate Now" else retire_age
    years_to_rot = target_age - age
    # Projected portfolio at rotation (exponential drift)
    proj = current_value * np.exp(btc_return * years_to_rot)
    rot_amt = proj * rotation_percent

    # Income over retirement
    ann_inc = rot_amt * (x[0]*msty_yield + x[1]*strk_yield + x[2]*strf_yield)
    years_ret = max(0, 90 - retire_age)
    cum_inc = ann_inc * years_ret

    # Simulate variance of capital at retirement
    caps = []
    for _ in range(200):
        val = proj * (1 - rotation_percent)
        val *= np.exp(btc_return * years_ret + volatility*np.random.randn()*np.sqrt(years_ret))
        caps.append(val)
    cap_var = np.var(caps)

    # Combined objective: maximize income, penalize variance
    alpha = inc_pref / 100
    score = alpha * cum_inc - (1 - alpha) * risk_aversion * cap_var
    return -score

res = minimize(score_alloc, [1/3,1/3,1/3],
               bounds=[(0,1)]*3,
               constraints=({'type':'eq','fun':lambda x: sum(x)-1},))
opt = res.x if res.success else [1/3,1/3,1/3]
msty_pct, strk_pct, strf_pct = [int(100*v) for v in opt]

st.subheader("🔀 Allocation Recommendation")
manual = st.checkbox("Manual Allocation")
if manual:
    msty_pct = st.slider("MSTY (%)", 0, 100, msty_pct)
    max_strk = 100 - msty_pct
    strk_pct = st.slider("STRK (%)", 0, max_strk, min(strk_pct, max_strk)) if max_strk > 0 else 0
    strf_pct = 100 - msty_pct - strk_pct
    if strf_pct < 0:
        st.error("Sum > 100%. Adjust sliders.")
        strf_pct = 0
else:
    st.markdown(f"- MSTY: **{msty_pct}%**, STRK: **{strk_pct}%**, STRF: **{strf_pct}%**")

st.progress((msty_pct + strk_pct + strf_pct) / 100)

# ---- PROJECTED INCOME & COMPARISON ---- #
def project_outcomes(rotation_age):
    yrs = rotation_age - age
    proj_val = current_value * np.exp(btc_return * yrs)
    rot_amt  = proj_val * rotation_percent
    yrs_ret  = max(0, 90 - rotation_age)
    cap_end  = rot_amt*(1-rotation_percent)*np.exp(btc_return * yrs_ret)
    ann_inc  = rot_amt*(msty_pct/100*msty_yield + strk_pct/100*strk_yield + strf_pct/100*strf_yield)
    cum_inc  = ann_inc * yrs_ret
    return cap_end, cum_inc

cap_now, inc_now = project_outcomes(age)
cap_ret, inc_ret = project_outcomes(retire_age)

st.header("🔍 Comparison")
c1, c2, c3 = st.columns(3)
with c1:
    st.write("**Metric**")
    st.write("Capital @90")
    st.write("Cum. Income")
with c2:
    st.write("**Rotate Now**")
    st.write(f"${cap_now:,.0f}")
    st.write(f"${inc_now:,.0f}")
with c3:
    st.write(f"**Rotate at {retire_age}**")
    st.write(f"${cap_ret:,.0f}")
    st.write(f"${inc_ret:,.0f}")

if (cap_now + inc_now) > (cap_ret + inc_ret):
    st.success("▶️ Rotating Now yields the best combined outcome.")
else:
    st.info(f"▶️ Waiting until age {retire_age} may yield a better outcome.")

# ---- TIMELINE ---- #
st.subheader("📅 Timeline")
fig, ax = plt.subplots(figsize=(8,2))
ax.axvline(age, color='blue', label="Today")
if rot_age is not None: ax.axvline(rot_age, color='green', label="Rotate")
ax.axvline(retire_age, color='gray', linestyle='--', label="Retire")
label = action if action != "Rotate Later" else f"Rotate at {retire_age}"
ax.text((age + (rot_age or retire_age)) / 2, 0.5, label, ha='center')
ax.set_xlim(age-1, retire_age+1)
ax.get_yaxis().set_visible(False)
ax.legend()
st.pyplot(fig)

# ---- MONTE CARLO OUTLOOK ---- #
st.subheader("📈 Outlook")
yrs1 = (rot_age or retire_age) - age
yrs2 = retire_age - (rot_age or retire_age)
sim = np.zeros((yrs1+yrs2+1, num_simulations))
sim[0] = current_value

for t in range(1, yrs1+1):
    sim[t] = sim[t-1]*np.exp((btc_return-0.5*volatility**2) + volatility*np.random.randn(num_simulations))
sim[yrs1] *= (1-rotation_percent)
for t in range(yrs1+1, yrs1+yrs2+1):
    sim[t] = sim[t-1]*np.exp((btc_return-0.5*volatility**2) + volatility*np.random.randn(num_simulations))

mean_path = sim.mean(axis=1)
ages = np.arange(age, retire_age+1)

fig2, ax2 = plt.subplots()
ax2.plot(ages, mean_path, label="Capital")
inc_series = [0]*yrs1 + [inc_now if action=="Rotate Now" else inc_ret]*(yrs2+1)
ax2.bar(ages, inc_series, alpha=0.4, label="Income")
for idx, val in enumerate(inc_series):
    if val>0:
        ax2.text(ages[idx], val, f"${val:,.0f}", ha='center', va='bottom', fontsize=8)
ax2.axvline(retire_age, color='gray', linestyle='--')
ax2.set_xlabel("Age")
ax2.set_ylabel("USD")
ax2.legend()
ax2.ticklabel_format(style='plain', axis='y')
st.pyplot(fig2)

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

    st.header("Core Signals & Preferences")
    sth_sopa   = st.number_input("STH-SOPA", 1.00, step=0.01)
    inc_pref   = st.slider("Income Preference (%)", 0, 100, 50)
    btc_return = st.slider("BTC Expected Annual Return (%)", 0.0, 50.0, 15.0, 0.1) / 100

    risk_choice = st.selectbox(
        "Risk Profile",
        ["Conservative", "Balanced", "Aggressive", "Degen"],
        index=1,
        help="How much to penalize portfolio variance when optimizing."
    )
    risk_aversion_map = {
        "Conservative": 1e-4,
        "Balanced":     5e-5,
        "Aggressive":   1e-5,
        "Degen":        0.0
    }
    risk_aversion = risk_aversion_map[risk_choice]

    with st.expander("Advanced Signals (optional)"):
        sth_mvrv_z = st.number_input("STH-MVRV-Z", 1.00, step=0.1)
        fund_rate  = st.number_input("Futures Funding Rate (%)", 2.00, step=0.01)

# ---- PORTFOLIO VALUE ---- #
current_value = mstr_price * shares
st.metric("💼 Portfolio Value", f"${current_value:,.0f}")

# ---- BAYESIAN PROBABILITY ---- #
data_pts = 100
data_pts += 50 if sth_sopa > 1 else -25 if sth_sopa < 1 else 0
if 'sth_mvrv_z' in locals():
    data_pts += -25 if sth_mvrv_z > 6 else 0
if 'fund_rate' in locals():
    data_pts += -25 if fund_rate > 0.1 else 0
data_pts = max(data_pts, 10)

age_frac  = np.clip(1 - (retire_age - age) / 30, 0, 1)
boost_pts = int(age_frac * 2) + (1 if current_value >= threshold else 0)
prior_succ = int(bayesian_prior * data_pts)
posterior  = (prior_succ + boost_pts + 1) / (data_pts + 2)

# ---- ROTATION DECISION ---- #
if posterior >= 0.60:
    action, rot_age = "Rotate Now", age
elif posterior >= 0.50:
    action, rot_age = "Rotate Later", None
else:
    action, rot_age = "Hold Until Retirement", retire_age

st.subheader("🔁 Decision")
color = "green" if posterior >= 0.6 else "orange" if posterior >= 0.5 else "red"
st.markdown(
    f"**Rotation Probability:** <span style='color:{color}'>**{posterior:.1%}**</span>",
    unsafe_allow_html=True
)
st.markdown(f"**Action:** **{action}**")
if rot_age is not None:
    st.markdown(f"**Rotation Age:** **{rot_age}**")

# ---- ALLOCATION OPTIMIZER (RISK-ADJUSTED) ---- #
def score_alloc(x):
    target    = age if action == "Rotate Now" else retire_age
    yrs_rot   = target - age
    proj_val  = current_value * np.exp(btc_return * yrs_rot)
    rot_amt   = proj_val * rotation_percent

    ann_inc  = rot_amt * (x[0]*msty_yield + x[1]*strk_yield + x[2]*strf_yield)
    yrs_ret  = max(0, 82 - retire_age)
    cum_inc  = ann_inc * yrs_ret

    caps = []
    for _ in range(200):
        val = proj_val * (1 - rotation_percent)
        val *= np.exp(btc_return * yrs_ret + volatility * np.random.randn() * np.sqrt(yrs_ret))
        caps.append(val)
    cap_var = np.var(caps)

    alpha = inc_pref / 100
    score = alpha * cum_inc - (1 - alpha) * risk_aversion * cap_var
    return -score

res = minimize(
    score_alloc,
    [1/3, 1/3, 1/3],
    bounds=[(0,1)]*3,
    constraints=({'type':'eq','fun':lambda x: sum(x)-1},)
)
opt = res.x if res.success else [1/3, 1/3, 1/3]
msty_pct, strk_pct, strf_pct = [int(100*v) for v in opt]

st.subheader("🔀 Allocation Recommendation")
manual = st.checkbox("Manual Allocation")
if manual:
    msty_pct = st.slider("MSTY (%)", 0, 100, msty_pct)
    max_strk = 100 - msty_pct
    strk_pct = st.slider("STRK (%)", 0, max_strk, min(strk_pct, max_strk)) if max_strk > 0 else 0
    strf_pct = 100 - msty_pct - strk_pct
    if strf_pct < 0:
        st.error("Total >100%. Adjust sliders.")
        strf_pct = 0
else:
    st.markdown(f"- MSTY: **{msty_pct}%**, STRK: **{strk_pct}%**, STRF: **{strf_pct}%**")

st.progress((msty_pct + strk_pct + strf_pct) / 100)

# ---- PROJECTED INCOME & COMPARISON ---- #
def project_outcomes(rotation_age):
    yrs      = rotation_age - age
    proj_val = current_value * np.exp(btc_return * yrs)
    rot_amt  = proj_val * rotation_percent
    yrs_ret  = max(0, 82 - rotation_age)
    cap_end  = rot_amt * (1 - rotation_percent) * np.exp(btc_return * yrs_ret)
    ann_inc  = rot_amt * (
        msty_pct/100*msty_yield + strk_pct/100*strk_yield + strf_pct/100*strf_yield
    )
    cum_inc = ann_inc * yrs_ret
    return cap_end, cum_inc

cap_now, inc_now = project_outcomes(age)
cap_ret, inc_ret = project_outcomes(retire_age)

st.header("🔍 Comparison")
c1, c2, c3 = st.columns(3)
with c1:
    st.write("**Metric**")
    st.write("Capital @82")
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
    st.info(f"▶️ Waiting until age {retire_age} may yield a better combined outcome.")

# ---- TIMELINE ---- #
st.subheader("📅 Timeline")
fig, ax = plt.subplots(figsize=(8,2))
ax.axvline(age, color='blue', label="Today")
if rot_age is not None: ax.axvline(rot_age, color='green', label="Rotate")
ax.axvline(retire_age, color='gray', linestyle='--', label="Retire")
label = action if rot_age is not None else f"Rotate at {retire_age}"
ax.text((age + (rot_age or retire_age)) / 2, 0.5, label, ha='center')
ax.set_xlim(age-1, retire_age+1)
ax.get_yaxis().set_visible(False)
ax.legend()
st.pyplot(fig)

# ---- CASH-FLOW OUTLOOK THROUGH AGE 82 ---- #
st.subheader("📈 Cash-Flow Outlook Through Age 82")
death_age = 82
yrs_to_rot = (rot_age or retire_age) - age
yrs_post   = death_age - (rot_age or retire_age)
sim        = np.zeros((yrs_to_rot+yrs_post+1, num_simulations))
sim[0]     = current_value

for t in range(1, yrs_to_rot+1):
    sim[t] = sim[t-1] * np.exp((btc_return - 0.5*volatility**2)
                                + volatility * np.random.randn(num_simulations))
for t in range(yrs_to_rot+1, yrs_to_rot+yrs_post+1):
    sim[t] = sim[yrs_to_rot]

mean_path = sim.mean(axis=1)
ages_all  = np.arange(age, death_age+1)
inc_all   = [0]*yrs_to_rot + [inc_now if action=="Rotate Now" else inc_ret]*(yrs_post+1)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(ages_all, mean_path, label="Capital (frozen post-rotation)", linewidth=2)
bars = ax.bar(ages_all, inc_all, alpha=0.4, label="Annual Income")
ax.bar_label(
    bars,
    labels=[f"${h:,.0f}" if h>0 else "" for h in inc_all],
    padding=3,
    fontsize=8,
    rotation=90,
    label_type='edge'
)
ax.axvline(rot_age or retire_age, color='green', linestyle='--', label="Rotation")
ax.axvline(retire_age, color='gray', linestyle='-.', label="Retirement")
ax.axvline(death_age, color='black', linestyle=':', label=f"Life Expectancy ({death_age})")
ax.set_xlabel("Age")
ax.set_ylabel("USD")
ax.legend(loc='upper left')
ax.ticklabel_format(style='plain', axis='y')
plt.tight_layout()
st.pyplot(fig)

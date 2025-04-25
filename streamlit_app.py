import streamlit as st
import numpy as np
import yfinance as yf
import pandas as pd
from scipy.optimize import minimize

# ---- SETTINGS ---- #
DEFAULT_SHARES   = 100
ROT_FRAC         = 0.20
NUM_SIM          = 500
YIELDS           = {'MSTY':0.15, 'STRK':0.07, 'STRF':0.07}

# ---- PAGE SETUP ---- #
st.set_page_config("MSTR Retirement Assistant", layout="wide")
st.title("📊 MSTR Retirement Decision Assistant")

# ---- LIVE PRICE ---- #
price = yf.Ticker("MSTR").history(period="1d")['Close'].iloc[-1]
st.metric("📈 MSTR Price", f"${price:,.2f}")

# ---- SIDEBAR INPUTS ---- #
with st.sidebar:
    st.header("Profile & Goals")
    shares      = st.number_input("Current Shares", value=DEFAULT_SHARES, step=1)
    years_ret   = st.slider("Years to Retirement", 1, 40, 7)
    target_inc  = st.number_input("Desired Annual Income ($)", 0, 500_000, 50_000, step=1_000)

    st.header("Assumptions")
    btc_ret     = st.slider("BTC Return (%)", 0.0, 50.0, 15.0, 0.1)/100
    keep_pct    = st.slider("Keep in MSTR (%)", 0, 100, 20)
    inc_pref    = st.slider("Income Preference (%)", 0, 100, 50)
    risk_choice = st.selectbox("Risk Profile", ["Conservative", "Balanced", "Aggressive", "Degen"])
    risk_map    = {"Conservative":1e-4, "Balanced":5e-5, "Aggressive":1e-5, "Degen":0.0}
    risk_av     = risk_map[risk_choice]

# ---- INCOME PLANNING ---- #
growth    = np.exp(btc_ret * years_ret)
blended_y = (YIELDS['MSTY'] + YIELDS['STRK'] + YIELDS['STRF']) / 3
denom     = price * growth * ROT_FRAC * blended_y
req_sh    = target_inc / denom if denom > 0 else 0

st.subheader("🎯 Income Planning")
st.write(f"- You hold **{shares:,}** shares today.")
st.write(f"- To generate **${target_inc:,.0f}/yr**, you need **{int(np.ceil(req_sh)):,}** shares.")

# ---- ALLOCATION OPTIMIZER ---- #
def objective(x):
    proj     = shares * price * growth
    eff_rot  = ROT_FRAC * (1 - keep_pct/100)
    kept     = proj * (keep_pct/100)
    yrs_post = years_ret
    kept    *= np.exp(btc_ret * yrs_post)
    ann_inc  = proj * eff_rot * (x[0]*YIELDS['MSTY'] + x[1]*YIELDS['STRK'] + x[2]*YIELDS['STRF'])
    cum_inc  = ann_inc * yrs_post
    # simple variance penalty on kept slice
    var      = np.var([kept * np.exp(0.1*np.random.randn()) for _ in range(200)])
    alpha    = inc_pref / 100
    return -(alpha * cum_inc + (1 - alpha) * kept - (1 - alpha) * risk_av * var)

res    = minimize(
    objective,
    [1/3, 1/3, 1/3],
    bounds=[(0,1)]*3,
    constraints={'type':'eq','fun':lambda x: sum(x)-1}
)
alloc  = res.x if res.success else [1/3, 1/3, 1/3]
m_pct, s_pct, f_pct = [int(100*v) for v in alloc]

st.subheader("🔀 Allocation Recommendation")
st.write(f"- MSTY: **{m_pct}%**, STRK: **{s_pct}%**, STRF: **{f_pct}%**")

# ---- COMPARISON TABLE ---- #
def project_outcome(sh):
    proj     = sh * price * growth
    eff_rot  = ROT_FRAC * (1 - keep_pct/100)
    kept     = proj * (keep_pct/100)
    yrs_post = years_ret
    kept    *= np.exp(btc_ret * yrs_post)
    cum_inc  = proj * eff_rot * (m_pct/100*YIELDS['MSTY'] + s_pct/100*YIELDS['STRK'] + f_pct/100*YIELDS['STRF']) * yrs_post
    return kept, cum_inc

kn, in_now = project_outcome(shares)
kr, in_req  = project_outcome(int(req_sh))

st.subheader("🔍 Comparison")
df = pd.DataFrame({
    "Metric":      ["Kept Value", "Cumulative Income"],
    "Current":     [f"${kn:,.0f}", f"${in_now:,.0f}"],
    "Required":    [f"${kr:,.0f}", f"${in_req:,.0f}"]
}).set_index("Metric")
st.table(df)

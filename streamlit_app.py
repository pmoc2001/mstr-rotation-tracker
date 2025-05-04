import streamlit as st
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import subprocess


# --- Automatically get current Git commit hash ---
def get_git_commit_hash():
    try:
        commit_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
    except Exception:
        commit_hash = "unknown"
    return commit_hash

VERSION = get_git_commit_hash()

# ---- PAGE CONFIG ---- #
st.set_page_config(page_title="MSTR Retirement Assistant", layout="wide")
st.title("📊 MSTR Retirement Decision Assistant")
st.caption(f"Git Commit Version: `{VERSION}`")

# ---- SETTINGS ---- #
default_shares = 1650
rotation_percent = 0.20
bayesian_prior = 0.515
num_simulations = 500
volatility = 0.7
life_expectancy = 82

# Real-world yields
msty_yield = 0.15
strk_yield = 0.07
strf_yield = 0.07

# ---- PAGE CONFIG ---- #
st.set_page_config(page_title="MSTR Retirement Assistant", layout="wide")
tabs = st.tabs(["Decision Tool", "Documentation"])
tool_tab, doc_tab = tabs

with tool_tab:
    st.title("📊 MSTR Retirement Decision Assistant")
    st.markdown("Helps decide **when** and **how** to rotate MSTR into income-producing assets.")

    # ---- LIVE PRICES ---- #
    btc_hist = yf.Ticker("BTC-USD").history(period="1y")
    btc_now = btc_hist['Close'].iloc[-1]
    btc_ath = btc_hist['Close'].max()
    btc_200dma = btc_hist['Close'].rolling(200).mean().iloc[-1]
    mstr_price = yf.Ticker("MSTR").history(period="1d")['Close'].iloc[-1]

    drawdown = (btc_now - btc_ath) / btc_ath
    above_200dma = btc_now > btc_200dma

    # ---- SIDEBAR ---- #
    with st.sidebar:
        shares = st.number_input("Shares Held", value=default_shares, step=10)
        age = st.number_input("Current Age", 48, 80, 48)
        retire_age = st.slider("Retirement Age", age+1, 80, age+7)
        threshold = st.selectbox("Rotation Threshold ($)", [600_000, 750_000, 1_000_000])

        st.header("Core Preferences")
        inc_pref = st.slider("Income Preference (%)", 0, 100, 50)
        btc_return = st.slider("BTC Expected Annual Return (%)", 0.0, 50.0, 15.0) / 100
        keep_mstr_pct = st.slider("Keep in MSTR (%)", 0, 100, 20)
        risk_choice = st.selectbox("Risk Profile", ["Conservative", "Balanced", "Aggressive", "Degen"])
        risk_aversion = {"Conservative":1e-4, "Balanced":5e-5, "Aggressive":1e-5, "Degen":0.0}[risk_choice]

    current_value = mstr_price * shares
    st.metric("💼 Current Portfolio", f"${current_value:,.0f}")

    # ---- DECISION SCORE (Simplified Weighted Scoring) ---- #
    market_sentiment = 0.5 * (1 + drawdown) + 0.5 * above_200dma
    age_factor = 1 - (retire_age - age)/30
    threshold_factor = min(current_value/threshold, 1.0)

    posterior = bayesian_prior + 0.3 * market_sentiment + 0.2 * (age_factor + threshold_factor)/2
    posterior = np.clip(posterior, 0, 1)

    action = ("Rotate Now" if posterior >= 0.60 else 
              "Rotate Later" if posterior >= 0.50 else 
              "Hold Until Retirement")
    rot_age = age if action == "Rotate Now" else retire_age

    st.subheader("🔁 Decision")
    st.markdown(f"**Rotation Probability:** {posterior:.1%} – **{action}**")

    # ---- ALLOCATION OPTIMIZER ---- #
    def score_alloc(x):
        proj_val = current_value * np.exp(btc_return * (rot_age - age))
        eff_rot = rotation_percent * (1 - keep_mstr_pct/100)
        kept_frac = rotation_percent * keep_mstr_pct/100
        yrs_post = life_expectancy - rot_age
        kept_val = proj_val * kept_frac * np.exp(btc_return * yrs_post)
        ann_inc = proj_val * eff_rot * (x[0]*msty_yield + x[1]*strk_yield + x[2]*strf_yield)
        cum_inc = ann_inc * yrs_post
        caps = [kept_val * np.exp(volatility*np.random.randn()*np.sqrt(yrs_post)) for _ in range(200)]
        cap_var = np.var(caps)
        alpha = inc_pref/100
        return -(alpha*cum_inc + (1-alpha)*kept_val - (1-alpha)*risk_aversion*cap_var)

    res = minimize(score_alloc, [0.34, 0.33, 0.33],
                   bounds=[(0,1)]*3,
                   constraints={'type':'eq','fun':lambda x:sum(x)-1})
    msty_pct, strk_pct, strf_pct = [int(100*v) for v in (res.x if res.success else [1/3]*3)]

    st.subheader("🔀 Allocation")
    if st.checkbox("Manual Allocation"):
        msty_pct = st.slider("MSTY (%)", 0, 100, msty_pct)
        remaining_pct = 100 - msty_pct
        strk_pct = st.slider("STRK (%)", 0, remaining_pct, min(strk_pct, remaining_pct))
        strf_pct = 100 - msty_pct - strk_pct
        st.write(f"STRF (%): {strf_pct}%")
    else:
        st.write(f"MSTY: {msty_pct}%, STRK: {strk_pct}%, STRF: {strf_pct}%")

    # ---- REALISTIC PROJECTIONS ---- #
    def project(age_proj):
        proj_val = current_value * np.exp(btc_return * (age_proj-age))
        eff_rot = rotation_percent * (1 - keep_mstr_pct/100)
        kept_frac = rotation_percent * keep_mstr_pct/100
        yrs_post = life_expectancy - age_proj
        kept_val = proj_val * kept_frac * np.exp(btc_return * yrs_post)
        ann_inc = proj_val * eff_rot * (msty_pct/100*msty_yield + strk_pct/100*strk_yield + strf_pct/100*strf_yield)
        return kept_val, ann_inc * yrs_post

    kv_now, ci_now = project(age)
    kv_ret, ci_ret = project(retire_age)

    st.subheader("🔍 Outcomes")
    st.table({
        "Metric": ["MSTR @82", "Cumulative Income"],
        "Rotate Now": [f"${kv_now:,.0f}", f"${ci_now:,.0f}"],
        "At Retirement": [f"${kv_ret:,.0f}", f"${ci_ret:,.0f}"]
    })

with doc_tab:
    st.title("📘 Documentation & Assumptions")
    st.markdown("""
    - Uses simplified weighted scoring method (not strictly Bayesian).
    - Annual yields: MSTY 15%, STRK 7%, STRF 7%.
    - Realistic outcomes shown, no inflation adjustments.
    - No tax/fees considered.
    - Investment horizon until age 82.
    """)


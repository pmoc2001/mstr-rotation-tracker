import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time
from datetime import date

# ---- SETTINGS ---- #
DEFAULT_SHARES   = 100
ROT_FRAC         = 0.20
BAYES_PRIOR      = 0.515
NUM_SIM          = 500
VOLATILITY       = 0.7
YIELDS           = {'MSTY':0.15, 'STRK':0.07, 'STRF':0.07}
LIFE_EXP         = 82

# ---- PAGE CONFIG ---- #
st.set_page_config("MSTR Retirement Assistant", layout="wide")
tabs = st.tabs(["Decision Tool", "Documentation"])
tool, docs = tabs

with tool:
    st.title("📊 MSTR Retirement Decision Assistant")

    # Live price
    price = yf.Ticker("MSTR").history(period="1d")['Close'].iloc[-1]
    st.metric("📈 MSTR Price", f"${price:,.2f}")

    # Sidebar inputs
    with st.sidebar:
        st.header("Profile & Goals")
        shares      = st.number_input("Current Shares", DEFAULT_SHARES, step=1)
        years_to_ret= st.slider("Years to Retirement", 1, 40, 7)
        target_inc  = st.number_input("Desired Annual Income ($)", 0, 1_000_000, 50_000, step=1_000)

        st.header("Assumptions")
        btc_ret     = st.slider("BTC Return (%)", 0.0, 50.0, 15.0, 0.1)/100
        keep_pct    = st.slider("Keep in MSTR (%)", 0, 100, 20)
        sth_sopa    = st.number_input("STH-SOPA", 1.00, step=0.01)
        inc_pref    = st.slider("Income Preference (%)", 0, 100, 50)
        risk_choice = st.selectbox("Risk Profile", ["Conservative","Balanced","Aggressive","Degen"])
        risk_map    = {"Conservative":1e-4,"Balanced":5e-5,"Aggressive":1e-5,"Degen":0.0}
        risk_av     = risk_map[risk_choice]

    # === Income Planning ===
    growth    = np.exp(btc_ret * years_to_ret)
    blended_y = (YIELDS['MSTY']+YIELDS['STRK']+YIELDS['STRF'])/3
    denom     = price * growth * ROT_FRAC * blended_y
    needed_sh = target_inc/denom if denom>0 else np.nan

    st.subheader("🎯 Income Planning")
    st.write(f"You hold **{shares:,}** shares today.")
    st.write(f"To generate **${target_inc:,.0f}/yr**, you need **{int(np.ceil(needed_sh)):,}** shares.")

    # === Bayesian Decision ===
    data_pts = 100 + (50 if sth_sopa>1 else -25 if sth_sopa<1 else 0)
    data_pts = max(data_pts, 10)
    boost    = int(np.clip(1 - years_to_ret/30, 0,1)*2) + (1 if shares*price>=0 else 0)
    post     = (int(BAYES_PRIOR*data_pts) + boost + 1)/(data_pts+2)
    action   = "Rotate Now" if post>=0.6 else "Rotate Later" if post>=0.5 else "Hold"
    st.subheader("🔁 Decision")
    st.markdown(f"**Rotation Probability:** **{post:.1%}** – **{action}**")

    with st.expander("🧮 Bayesian Debug"):
        st.write(f"Data Points: {data_pts}")
        st.write(f"Boost: {boost}")
        st.write(f"Posterior: {post:.3f}")

    # === Allocation Optimizer ===
    def objective(x):
        proj    = shares*price*growth
        eff     = ROT_FRAC*(1-keep_pct/100)
        kept    = proj*(keep_pct/100)
        yrs_post= LIFE_EXP - years_to_ret
        kept   *= np.exp(btc_ret*yrs_post)
        inc_ann = proj*eff*(x[0]*YIELDS['MSTY']+x[1]*YIELDS['STRK']+x[2]*YIELDS['STRF'])
        cum_inc = inc_ann*yrs_post
        var     = np.var([kept*np.exp(VOLATILITY*np.random.randn()*np.sqrt(yrs_post)) for _ in range(200)])
        alpha   = inc_pref/100
        return -(alpha*cum_inc + (1-alpha)*kept - (1-alpha)*risk_av*var)

    res   = minimize(objective, [1/3]*3, bounds=[(0,1)]*3, constraints={'type':'eq','fun': lambda x: sum(x)-1})
    alloc = res.x if res.success else [1/3]*3
    m_pct, s_pct, f_pct = [int(100*v) for v in alloc]

    st.subheader("🔀 Allocation")
    st.write(f"- MSTY: {m_pct}%, STRK: {s_pct}%, STRF: {f_pct}%")

    # === Comparison ===
    def project(sh):
        proj    = sh*price*growth
        eff     = ROT_FRAC*(1-keep_pct/100)
        kept    = proj*(keep_pct/100)
        yrs_post= LIFE_EXP - years_to_ret
        kept   *= np.exp(btc_ret*yrs_post)
        inc_tot = proj*eff*(m_pct/100*YIELDS['MSTY']+s_pct/100*YIELDS['STRK']+f_pct/100*YIELDS['STRF'])*yrs_post
        return kept, inc_tot

    kept_now, inc_now = project(shares)
    kept_req, inc_req = project(int(needed_sh))

    st.subheader("🔍 Comparison")
    df = pd.DataFrame({
        "Metric": ["Kept @82", "Cum. Income"],
        "Current": [f"${kept_now:,.0f}", f"${inc_now:,.0f}"],
        "Required": [f"${kept_req:,.0f}", f"${inc_req:,.0f}"]
    }).set_index("Metric")
    st.table(df)

    # === Timeline ===
    yrs_pre  = years_to_ret
    yrs_post = LIFE_EXP - years_to_ret
    fig, ax = plt.subplots(figsize=(8,2))
    ax.axvline(yrs_pre, color='gray', linestyle='--', label="Retirement")
    ax.legend()
    st.pyplot(fig)

    # === Monte Carlo Outlook ===
    st.subheader("📈 Monte Carlo Outlook")
    sim = np.zeros((yrs_pre+yrs_post+1, NUM_SIM))
    sim[0] = shares * price
    for t in range(1, yrs_pre+1):
        sim[t] = sim[t-1] * np.exp((btc_ret-0.5*VOLATILITY**2)+VOLATILITY*np.random.randn(NUM_SIM))
    sim[yrs_pre] *= (1-ROT_FRAC)
    for t in range(yrs_pre+1, yrs_pre+yrs_post+1):
        sim[t] = sim[t-1] * np.exp((btc_ret-0.5*VOLATILITY**2)+VOLATILITY*np.random.randn(NUM_SIM))
    st.line_chart(sim.mean(axis=1))

with docs:
    st.header("📘 Documentation")
    st.markdown("""
    **Features:**
    - Income Planning: shares needed for a target income.
    - Bayesian Decision: STH-SOPA driven rotate-now vs later.
    - Allocation Optimizer: splits to MSTY/STRK/STRF.
    - Comparison Table: kept vs required outcomes.
    - Timeline & Monte Carlo Outlook.

    **Assumptions & Defaults**  
    - BTC Return: 15% p.a.  
    - Yields: MSTY 15%, STRK/STRF 7%  
    - Rotation Fraction: 20%  
    - Life Expectancy: 82 years  
    """)

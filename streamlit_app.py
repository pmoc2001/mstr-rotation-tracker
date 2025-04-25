import streamlit as st
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from datetime import date

# ---- DEFAULTS ---- #
DEFAULT_SHARES     = 100
ROTATION_FRAC      = 0.20
BTC_RETURN_DEFAULT = 0.15  # 15%
YIELDS             = {'MSTY':0.15, 'STRK':0.07, 'STRF':0.07}

# ---- PAGE SETUP ---- #
st.set_page_config("MSTR Retirement Assistant", layout="wide")
st.title("📊 MSTR Retirement Decision Assistant")

# ---- SIDEBAR INPUTS ---- #
with st.sidebar:
    st.header("Profile & Goals")
    shares       = st.number_input("Current MSTR Shares", value=DEFAULT_SHARES, step=1)
    retire_years = st.slider("Years Until Retirement", 1, 40, 7)
    target_income= st.number_input("Desired Annual Income ($)", 0, 1_000_000, 50_000, step=1_000)

    st.header("Assumptions")
    btc_return   = st.slider("BTC Annual Return (%)", 0.0, 50.0, BTC_RETURN_DEFAULT*100, step=0.1)/100
    keep_pct     = st.slider("Keep in MSTR (%)", 0, 100, 20)
    inc_pref     = st.slider("Income Preference (%)", 0, 100, 50)
    risk_profile = st.selectbox("Risk Profile", ["Conservative","Balanced","Aggressive","Degen"])
    risk_map     = {"Conservative":1e-4,"Balanced":5e-5,"Aggressive":1e-5,"Degen":0.0}
    risk_aversion= risk_map[risk_profile]

# ---- LIVE PRICE ---- #
price = yf.Ticker("MSTR").history(period="1d")['Close'].iloc[-1]
st.metric("📈 MSTR Price", f"${price:,.2f}")

# ---- CALCULATE SHARES FOR INCOME ---- #
# Blended yield = assume rotating all ROTATION_FRAC of portfolio yields Y
Y = ROTATION_FRAC * (
    (YIELDS['MSTY'] + YIELDS['STRK'] + YIELDS['STRF']) / 3
)
growth = price * np.exp(btc_return * retire_years)
den = growth * ROTATION_FRAC * Y
shares_needed = target_income / den if den>0 else np.nan

st.subheader("🎯 Income Planning")
st.write(f"- You hold **{shares}** shares today.")
st.write(f"- To generate **${target_income:,.0f}/yr**, you need **{int(np.ceil(shares_needed)):,}** shares.")

# ---- SIMPLE DECISION OUTPUT ---- #
st.subheader("🔁 Quick Decision Check")
# e.g., if shares_needed > shares: suggest buy, else rotate
if shares < shares_needed:
    st.warning("▶️ You need more shares to meet your income goal.")
else:
    st.success("▶️ You have enough shares to meet your income goal.")

# ---- DOCUMENTATION ---- #
with st.expander("📘 Documentation"):
    st.markdown("""
    **Core Formula**  
    Required Shares = Target Income / (Price × e^(rT) × Rotation Fraction × Blended Yield)

    - **Price**: Current MSTR price  
    - **r**: BTC return assumption  
    - **T**: Years until retirement  
    - **Rotation Fraction**: Portion of portfolio rotated to income assets  
    - **Blended Yield**: Average of MSTY, STRK, STRF yields  

    This simplified tool focuses on your income goal. We can re-introduce the Bayesian decision, allocation optimizer, and Monte Carlo charts once this base is solid.
    """)

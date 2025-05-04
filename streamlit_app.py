import streamlit as st
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import subprocess

# ---- PAGE CONFIG ---- #
st.set_page_config(page_title="MSTR Retirement Assistant", layout="wide")

# --- Git Version ---
def get_git_commit_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
    except:
        return "unknown"
VERSION = get_git_commit_hash()

st.title("📊 MSTR Retirement Decision Assistant")
st.caption(f"Git Commit Version: `{VERSION}`")

# ---- SETTINGS ---- #
default_shares = 1650
rotation_percent = 0.20
btc_return = 0.15
life_expectancy = 82
msty_yield, strk_yield, strf_yield = 0.15, 0.07, 0.07
tax_relief_rate = 0.40  # Higher rate taxpayer as default

# ---- INPUTS SIDEBAR ---- #
with st.sidebar:
    st.header("👤 Profile & Goals")
    age = st.number_input("Current Age", 40, 70, 48)
    retire_age = st.slider("Retirement Age", age+1, 75, age+7)
    salary_monthly = st.number_input("Monthly Salary (£)", 1000, 20000, 5000, step=100)
    desired_income = st.number_input("Desired Retirement Income (£)", 10000, 150000, 50000, step=1000)

    st.header("📌 Portfolio & Assumptions")
    shares = st.number_input("MSTR Shares Held", 0, 5000, default_shares, step=10)
    keep_mstr_pct = st.slider("Keep in MSTR (%)", 0, 100, 20)

# ---- LIVE MARKET DATA ---- #
mstr_price = yf.Ticker("MSTR").history('1d')['Close'].iloc[-1]
portfolio_value = mstr_price * shares

st.metric("💼 Current MSTR Portfolio Value", f"${portfolio_value:,.0f}")

# ---- TAX OPTIMIZATION: SIPP CONTRIBUTIONS ---- #
st.header("💡 Pension Contribution Optimizer (UK SIPP)")
monthly_contrib = st.slider("Monthly Pension Contribution (£)", 0, int(salary_monthly), int(salary_monthly*0.15), step=100)

annual_contrib = monthly_contrib * 12
tax_relief = annual_contrib * tax_relief_rate
total_invested_annual = annual_contrib + tax_relief

st.markdown(f"""
- **Annual Pension Contribution:** £{annual_contrib:,.0f}
- **Annual Tax Relief (at {tax_relief_rate*100:.0f}%):** £{tax_relief:,.0f}
- **Total Annual Investment:** **£{total_invested_annual:,.0f}**
""")

years_to_retirement = retire_age - age
future_value = np.fv(rate=btc_return, nper=years_to_retirement, pmt=-total_invested_annual, pv=0)

st.metric("Projected Pension Pot at Retirement", f"£{future_value:,.0f}")

# ---- ALLOCATION OPTIMIZER ---- #
def allocation_objective(x):
    projected_portfolio = portfolio_value * np.exp(btc_return * years_to_retirement)
    rotate_amount = projected_portfolio * rotation_percent * (1 - keep_mstr_pct / 100)
    annual_income = rotate_amount * (x[0]*msty_yield + x[1]*strk_yield + x[2]*strf_yield)
    return -annual_income

res = minimize(allocation_objective, [0.34, 0.33, 0.33],
               bounds=[(0, 1)]*3, constraints={'type': 'eq', 'fun': lambda x: sum(x)-1})
msty_pct, strk_pct, strf_pct = [int(100*v) for v in res.x]

st.subheader("🔀 Optimal Allocation for Income Products")
st.write(f"- MSTY: **{msty_pct}%**, STRK: **{strk_pct}%**, STRF: **{strf_pct}%**")

# ---- PROJECTED INCOME & COMPARISON ---- #
def project(ret_age):
    proj_val = portfolio_value * np.exp(btc_return * (ret_age-age))
    eff_rot = rotation_percent*(1-keep_mstr_pct/100)
    yrs_post = life_expectancy - ret_age
    rot_amt = proj_val * eff_rot
    annual_income = rot_amt * (msty_pct/100*msty_yield + strk_pct/100*strk_yield + strf_pct/100*strf_yield)
    cumulative_income = annual_income * yrs_post
    return annual_income, cumulative_income

ai_now, ci_now = project(age)
ai_ret, ci_ret = project(retire_age)

st.subheader("🔍 Retirement Income Outcomes")
st.table({
    "Metric": ["Annual Income", "Cumulative Income to 82"],
    "If Rotated Now": [f"${ai_now:,.0f}", f"${ci_now:,.0f}"],
    f"At Age {retire_age}": [f"${ai_ret:,.0f}", f"${ci_ret:,.0f}"]
})

# ---- GRAPHICAL CASH-FLOW PROJECTION ---- #
st.subheader("📈 Cash-Flow Projection")
years = np.arange(age, life_expectancy+1)
portfolio_vals = portfolio_value * np.exp(btc_return * (years - age))

fig, ax = plt.subplots(figsize=(10,4))
ax.plot(years, portfolio_vals, label="Portfolio Value", color='blue')

annual_incomes = [0 if yr < retire_age else ai_ret for yr in years]
ax2 = ax.twinx()
ax2.bar(years, annual_incomes, color='orange', alpha=0.5, width=0.8, label='Annual Income')

ax.set_xlabel("Age")
ax.set_ylabel("Portfolio Value (£)", color='blue')
ax2.set_ylabel("Annual Income (£)", color='orange')
ax.tick_params(axis='y', labelcolor='blue')
ax2.tick_params(axis='y', labelcolor='orange')

ax.axvline(retire_age, linestyle='--', color='gray', label='Retirement Age')
ax.axvline(life_expectancy, linestyle=':', color='black', label='Life Expectancy (82)')

fig.tight_layout()
fig.legend(loc="upper left", bbox_to_anchor=(0.1,0.9))
st.pyplot(fig)

# ---- DOCUMENTATION TAB ---- #
st.markdown("---")
st.subheader("📖 Documentation & Assumptions")
st.markdown("""
- **Tax Relief Assumption:** Higher rate (40%) UK taxpayer.
- **Returns Assumption:** Annual BTC return at 15%.
- **Income Products:** MSTY (15%), STRK (7%), STRF (7%) yields.
- **No inflation adjustments or taxes/fees considered.**
- **Life Expectancy:** Calculations based on age 82.
""")

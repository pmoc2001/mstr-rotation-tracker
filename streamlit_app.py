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
default_shares = 100
rotation_percent = 0.20
btc_return = 0.15
life_expectancy = 82
msty_yield, strk_yield, strf_yield = 0.15, 0.07, 0.07

# ---- INPUTS SIDEBAR ---- #
with st.sidebar:
    st.header("👤 Profile & Goals")
    age = st.number_input("Current Age", 40, 70, 48)
    retire_age = st.slider("Retirement Age", age+1, 75, age+7)
    salary_monthly = st.number_input("Monthly Salary (£)", 1000, 25000, 10000, step=500)
    desired_income = st.number_input("Desired Retirement Income (£)", 10000, 150000, 50000, step=1000)

    st.header("📌 Portfolio")
    shares = st.number_input("MSTR Shares Held", 0, 5000, default_shares, step=10)
    keep_mstr_pct = st.slider("Keep in MSTR (%)", 0, 100, 20)

# ---- LIVE MARKET DATA ---- #
mstr_price = yf.Ticker("MSTR").history('1d')['Close'].iloc[-1]
portfolio_value = mstr_price * shares
st.metric("💼 Current MSTR Portfolio Value", f"${portfolio_value:,.0f}")

# ---- TAX OPTIMIZATION: £100k TAX TRAP ---- #
st.header("🧮 £100k Tax Trap Optimizer")

annual_salary = salary_monthly * 12
personal_allowance = 12570
allowance_reduction_threshold = 100000
effective_tax_trap_limit = 125140

def optimal_sipp_contrib(salary):
    if salary <= allowance_reduction_threshold:
        return 0
    required_reduction = salary - allowance_reduction_threshold
    optimal_contribution = min(required_reduction, salary - effective_tax_trap_limit)
    return optimal_contribution

opt_sipp_annual = optimal_sipp_contrib(annual_salary)
monthly_opt_sipp = opt_sipp_annual / 12

st.markdown(f"""
- **Annual Salary:** £{annual_salary:,.0f}
- **Optimal Annual SIPP Contribution to Avoid Trap:** **£{opt_sipp_annual:,.0f}** (£{monthly_opt_sipp:,.0f}/month)
- **Effective Tax Savings:** **£{opt_sipp_annual * 0.60:,.0f}** *(approx. 60% combined relief due to regained allowance)*
""")

monthly_contrib = st.slider("Adjust Monthly Pension Contribution (£)", 0, int(salary_monthly), int(monthly_opt_sipp), step=100)

annual_contrib = monthly_contrib * 12
total_tax_relief = annual_contrib * 0.40
total_invested_annual = annual_contrib + total_tax_relief

years_to_retirement = retire_age - age
future_value = total_invested_annual * (((1 + btc_return)**years_to_retirement - 1) / btc_return)

st.metric("Projected Pension Pot at Retirement", f"£{future_value:,.0f}")

# ---- ALLOCATION SLIDERS ---- #
st.header("🔀 Allocation Sliders")
col1, col2, col3 = st.columns(3)

with col1:
    msty_pct = st.slider("MSTY (%)", 0, 100, 50)
remaining_pct = 100 - msty_pct

with col2:
    strk_pct = st.slider("STRK (%)", 0, remaining_pct, remaining_pct // 2)
strf_pct = 100 - msty_pct - strk_pct

with col3:
    st.markdown(f"**STRF (%)**: {strf_pct}%")

st.progress((msty_pct+strk_pct+strf_pct)/100)

# ---- RETIREMENT INCOME PROJECTIONS ---- #
def project(ret_age):
    proj_val = portfolio_value * np.exp(btc_return * (ret_age - age))
    eff_rot = rotation_percent * (1 - keep_mstr_pct / 100)
    yrs_post = life_expectancy - ret_age
    rot_amt = proj_val * eff_rot
    annual_income = rot_amt * (msty_pct/100*msty_yield + strk_pct/100*strk_yield + strf_pct/100*strf_yield)
    cumulative_income = annual_income * yrs_post
    return annual_income, cumulative_income

ai_now, ci_now = project(age)
ai_ret, ci_ret = project(retire_age)

st.subheader("📉 Retirement Income Outcomes")
st.table({
    "Metric": ["Annual Income", "Cumulative Income to 82"],
    "Rotate Now": [f"${ai_now:,.0f}", f"${ci_now:,.0f}"],
    f"At Age {retire_age}": [f"${ai_ret:,.0f}", f"${ci_ret:,.0f}"]
})

# ---- GRAPHICAL CASH-FLOW PROJECTION ---- #
years = np.arange(age, life_expectancy+1)
portfolio_vals = portfolio_value * np.exp(btc_return * (years - age))
annual_incomes = [0 if yr < retire_age else ai_ret for yr in years]

fig, ax1 = plt.subplots(figsize=(10,4))
ax1.plot(years, portfolio_vals, label="Portfolio Value (£)", color='blue')
ax1.set_xlabel("Age")
ax1.set_ylabel("Portfolio Value (£)", color='blue')
ax1.tick_params(axis='y', colors='blue')

ax2 = ax1.twinx()
bars = ax2.bar(years, annual_incomes, color='orange', alpha=0.5, label='Annual Income (£)')
ax2.set_ylabel("Annual Income (£)", color='orange')
ax2.tick_params(axis='y', colors='orange')
ax2.bar_label(bars, padding=3, fontsize=8, rotation=90)

ax1.axvline(retire_age, linestyle='--', color='grey', label='Retirement Age')
ax1.axvline(life_expectancy, linestyle=':', color='black', label='Life Expectancy (82)')

fig.tight_layout()
fig.legend(loc="upper left", bbox_to_anchor=(0.1,0.9))
st.pyplot(fig)

# ---- DOCUMENTATION TAB ---- #
st.markdown("---")
st.subheader("📖 Documentation & Assumptions")
st.markdown("""
- **£100k tax trap modeling included** (UK specific).
- **Annual BTC growth assumption:** 15%.
- **Income yields:** MSTY (15%), STRK (7%), STRF (7%).
- **Inflation, additional taxes/fees not modeled explicitly.**
- **Life expectancy used:** Age 82.
- **Tax relief assumes higher rate (40%) taxpayer.**
""")

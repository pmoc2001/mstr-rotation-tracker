import streamlit as st
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import subprocess

# ---- PAGE CONFIG ---- #
st.set_page_config(page_title="MSTR Retirement Assistant", layout="wide")

# Git Version
def get_git_commit_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode().strip()
    except:
        return "unknown"
VERSION = get_git_commit_hash()

st.title("📊 MSTR Retirement Decision Assistant")
st.caption(f"Version: `{VERSION}`")

# ---- SETTINGS ---- #
default_shares = 100
btc_return = 0.15
life_expectancy = 82

# Updated yields based on verified data
msty_yield, strk_yield, strf_yield = 0.0174, 0.089, 0.1175

# ---- INPUTS SIDEBAR ---- #
with st.sidebar:
    st.header("👤 Profile & Goals")
    age = st.number_input("Current Age", 40, 70, 48)
    retire_age = st.slider("Retirement Age", age+1, 75, age+7)
    monthly_salary = st.number_input("Monthly Salary (£)", 1000, 25000, 10000, step=500)
    desired_income = st.number_input("Desired Retirement Income (£)", 10000, 150000, 50000, step=1000)

    st.header("📌 Portfolio")
    shares = st.number_input("MSTR Shares Held", 0, 5000, default_shares, step=10)
    keep_mstr_pct = st.slider("Keep in MSTR (%)", 0, 100, 20)

# ---- LIVE MARKET DATA ---- #
mstr_price = yf.Ticker("MSTR").history('1d')['Close'].iloc[-1]
portfolio_value = mstr_price * shares
st.metric("💼 Current MSTR Portfolio Value", f"${portfolio_value:,.0f}")

# ---- TAX TRAP CALCULATION ---- #
st.header("🧮 £100k Tax Trap Explained (UK)")

annual_salary = monthly_salary * 12
threshold = 100000
trap_limit = 125140

def optimal_sipp_contribution(salary):
    if salary <= threshold:
        return 0
    return min(salary - threshold, salary - trap_limit)

opt_sipp = optimal_sipp_contribution(annual_salary)
monthly_opt_sipp = opt_sipp / 12
effective_savings = opt_sipp * 0.6  # Effective 60% relief

st.markdown(f"""
- **Annual Salary:** £{annual_salary:,.0f}  
- **Optimal SIPP Contribution:** £{opt_sipp:,.0f} (£{monthly_opt_sipp:,.0f}/month)  
- **Estimated Effective Tax Savings:** £{effective_savings:,.0f} *(approx. 60% due to regained allowance)*

> 💡 **Plain English:**  
> If you earn over £100,000, your personal allowance is reduced, effectively taxing income between £100,000-£125,140 at 60%. Contributing at least **£{opt_sipp:,.0f}** annually (£{monthly_opt_sipp:,.0f}/month) into your SIPP reduces your taxable income back to £100,000, restoring your allowance and significantly lowering your taxes.
""")

monthly_contrib = st.slider("Monthly Pension Contribution (£)", 0, int(monthly_salary), int(monthly_opt_sipp), step=100)

annual_contrib = monthly_contrib * 12
tax_relief = annual_contrib * 0.4
total_annual_investment = annual_contrib + tax_relief
years_to_retirement = retire_age - age
future_sipp_value = total_annual_investment * (((1 + btc_return)**years_to_retirement - 1) / btc_return)

st.metric("Projected Pension Pot at Retirement", f"£{future_sipp_value:,.0f}")

# ---- ALLOCATION SLIDERS ---- #
st.header("🔀 Allocate Retirement Income")
msty_pct = st.slider("MSTY (%)", 0, 100, 50)
strk_pct = st.slider("STRK (%)", 0, 100 - msty_pct, (100 - msty_pct)//2)
strf_pct = 100 - msty_pct - strk_pct

st.markdown(f"**MSTY:** {msty_pct}% | **STRK:** {strk_pct}% | **STRF:** {strf_pct}%")
st.progress((msty_pct + strk_pct + strf_pct)/100)

# ---- INCOME PROJECTION ---- #
def income_projection(shares_held):
    future_value = (shares_held * mstr_price) * np.exp(btc_return * years_to_retirement)
    rotation_amount = future_value * (1 - keep_mstr_pct/100)
    annual_income = rotation_amount * (msty_pct/100*msty_yield + strk_pct/100*strk_yield + strf_pct/100*strf_yield)
    return annual_income

projected_income = income_projection(shares)
total_income_with_sipp = projected_income + (future_sipp_value * 0.04)  # 4% withdrawal from pension

gap = desired_income - total_income_with_sipp

st.subheader("🔍 Desired Income Analysis")
st.markdown(f"""
- Desired Annual Retirement Income: £{desired_income:,.0f}  
- Projected Retirement Income (MSTR + SIPP): £{total_income_with_sipp:,.0f}  

**Income Shortfall:** £{gap:,.0f}
""")

if gap > 0:
    additional_shares_needed = gap / (mstr_price * np.exp(btc_return * years_to_retirement) * rotation_percent * (msty_pct/100*msty_yield + strk_pct/100*strk_yield + strf_pct/100*strf_yield))
    additional_monthly_sipp = gap / ((future_sipp_value/years_to_retirement)*0.04)
    
    st.markdown(f"""
    ⚠️ **You have a retirement income gap. Consider:**  
    - Buying approx. **{int(additional_shares_needed)} more MSTR shares** now, or  
    - Increasing monthly pension contributions by **£{int(additional_monthly_sipp):,}/month**, or  
    - Delaying retirement.
    """)
else:
    st.success("🎉 You're on track to meet your retirement goals!")

# ---- GRAPHICAL PROJECTION ---- #
years = np.arange(age, life_expectancy+1)
portfolio_growth = portfolio_value * np.exp(btc_return * (years - age))

fig, ax1 = plt.subplots(figsize=(10,4))
ax1.plot(years, portfolio_growth, label="Portfolio Value (£)", color='blue')
ax1.set_xlabel("Age")
ax1.set_ylabel("Portfolio Value (£)", color='blue')

annual_incomes = [0 if yr < retire_age else total_income_with_sipp for yr in years]
ax2 = ax1.twinx()
bars = ax2.bar(years, annual_incomes, color='orange', alpha=0.5, label='Annual Income (£)')
ax2.set_ylabel("Annual Income (£)", color='orange')
ax2.bar_label(bars, padding=3, fontsize=8, rotation=90)

ax1.axvline(retire_age, linestyle='--', color='grey', label='Retirement Age')
ax1.axvline(life_expectancy, linestyle=':', color='black', label='Life Expectancy (82)')
fig.tight_layout()
fig.legend(loc="upper left", bbox_to_anchor=(0.1,0.9))
st.pyplot(fig)

# ---- DOCUMENTATION ---- #
st.subheader("📖 Documentation")
st.markdown("""
- **£100k Tax Trap clearly modeled and explained.**
- Real-world yields: MSTY (1.74%), STRK (8.9%), STRF (11.75%).
- Assumes 15% annual BTC growth.
- Pension withdrawal rate set at conservative 4%.
- Inflation, fees, and other taxes not included explicitly.
""")

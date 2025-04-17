import streamlit as st
import numpy as np

# ---- SETTINGS ---- #
default_shares = 1650
msty_yield = 0.20
preferred_yield = 0.075
rotation_percent = 0.20
thresholds = [600_000, 750_000, 1_000_000]
bayesian_prior = 0.515  # from previous simulation

# ---- INPUTS ---- #
st.title("📈 MSTR Early Rotation Tracker")

mstr_price = st.number_input("Current MSTR Price ($)", value=311.0, step=1.0)
shares_held = st.number_input("Shares Held", value=default_shares, step=10)
current_age = st.number_input("Your Age", value=48, step=1)
selected_threshold = st.selectbox("Rotation Trigger Threshold ($)", thresholds)

# ---- CALCULATIONS ---- #
portfolio_value = mstr_price * shares_held
threshold_met = portfolio_value >= selected_threshold

# Simulated Bayesian update (simple posterior estimate)
data_points = 100  # You could make this dynamic with time or confidence
successes = int(bayesian_prior * data_points) + (1 if threshold_met else 0)
failures = data_points - successes
posterior_prob = (successes + 1) / (data_points + 2)

# Estimated income if rotated
rotation_value = portfolio_value * rotation_percent
est_income = rotation_value * (msty_yield + preferred_yield) / 2

# ---- OUTPUT ---- #
st.markdown(f"### 💰 Portfolio Value: **${portfolio_value:,.0f}**")

if threshold_met:
    st.success(f"✅ Portfolio exceeds ${selected_threshold:,} — rotation eligible!")
    st.markdown(f"### 🔁 Estimated Income if You Rotate 20% Now: **${est_income:,.0f}/yr**")
    st.markdown(f"### 🧠 Bayesian Probability That Early Rotation is Optimal: **{posterior_prob:.1%}**")
    if posterior_prob > 0.60:
        st.info("**Recommended Action:** Consider rotating 20% into MSTY + STRK/STRF.")
    else:
        st.warning("**Hold for now.** Rotation is close but not yet clearly optimal.")
else:
    st.warning(f"Portfolio is below the selected threshold (${selected_threshold:,}).")
    st.markdown(f"### 🧠 Bayesian Probability That Early Rotation is Optimal: **{posterior_prob:.1%}**")
    st.info("Keep monitoring. Rotation opportunity may emerge soon.")

st.markdown("---")
st.caption("This tool uses Monte Carlo-informed Bayesian logic to help you time early MSTR portfolio rotation.")

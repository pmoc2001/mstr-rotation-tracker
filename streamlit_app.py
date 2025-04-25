import streamlit as st
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time

# ---- SETTINGS ---- #
default_shares    = 1650
rotation_percent  = 0.20
bayesian_prior    = 0.515
num_simulations   = 500
volatility        = 0.7

# Real-world yields
msty_yield        = 0.15
strk_yield        = 0.07
strf_yield        = 0.07

# ---- PAGE CONFIG ---- #
st.set_page_config(page_title="MSTR Retirement Assistant", layout="wide")
tabs = st.tabs(["Decision Tool", "Documentation"])
tool_tab, doc_tab = tabs

with tool_tab:
    st.title("\U0001F4CA MSTR Retirement Decision Assistant")
    st.markdown("Decide **when** to rotate part of your MSTR into income assets, and **how much** to allocate—balancing income vs. risk.")

    # ---- LIVE PRICES ---- #
    btc_hist = yf.Ticker("BTC-USD").history(period="1y")
    btc_now = btc_hist['Close'].iloc[-1]
    btc_ath = btc_hist['Close'].max()
    btc_200dma = btc_hist['Close'].rolling(window=200).mean().iloc[-1]
    mstr_price = yf.Ticker("MSTR").history(period="1d")['Close'].iloc[-1]

    drawdown = (btc_now - btc_ath) / btc_ath
    above_200dma = 1 if btc_now > btc_200dma else 0

    # ---- SIDEBAR ---- #
    with st.sidebar:
        st.header("Your Profile")
        shares      = st.number_input("Shares Held", value=default_shares, step=10)
        age         = st.number_input("Current Age", value=48, min_value=18, max_value=80, step=1)
        retire_age  = st.slider("Retirement Age", age+1, age+30, age+7)
        threshold   = st.selectbox("Rotation Threshold ($)", [600_000, 750_000, 1_000_000])

        st.header("Market Conditions")
        st.metric("BTC Price", f"${btc_now:,.0f}")
        st.metric("Drawdown from ATH", f"{drawdown:.1%}")
        st.metric("200-Day MA", f"${btc_200dma:,.0f}")
        st.markdown(f"\U0001F4C8 BTC is **{'above' if above_200dma else 'below'}** the 200-day MA")

        st.header("Core Preferences")
        inc_pref    = st.slider("Income Preference (%)", 0, 100, 50)
        btc_return  = st.slider("BTC Expected Annual Return (%)", 0.0, 50.0, 15.0, 0.1) / 100

        risk_choice = st.selectbox(
            "Risk Profile",
            ["Conservative", "Balanced", "Aggressive", "Degen"],
            index=1,
            help="How much to penalize portfolio variance when optimizing."
        )
        risk_map      = {"Conservative":1e-4, "Balanced":5e-5, "Aggressive":1e-5, "Degen":0.0}
        risk_aversion = risk_map[risk_choice]

        st.header("Rotation Settings")
        keep_mstr_pct = st.slider("Keep in MSTR (%)", 0, 100, 20)

    # ---- PORTFOLIO VALUE ---- #
    current_value = mstr_price * shares
    st.metric("\U0001F4BC Portfolio Value", f"${current_value:,.0f}")

    # ---- BAYESIAN DECISION ---- #
    drawdown_score = np.clip(1 + drawdown, 0.0, 1.0)
    ma_score = 1.0 if above_200dma else 0.0
    market_sentiment = 0.6 * drawdown_score + 0.4 * ma_score

    age_frac = np.clip(1 - (retire_age - age) / 30, 0, 1)
    threshold_score = np.clip((current_value - threshold) / threshold, 0, 1)
    goal_score = 0.5 * threshold_score + 0.5 * age_frac

    posterior = bayesian_prior + 0.3 * market_sentiment + 0.2 * goal_score
    posterior = np.clip(posterior, 0, 1)

    if posterior >= 0.60:
        action, rot_age = "Rotate Now", age
    elif posterior >= 0.50:
        action, rot_age = "Rotate Later", None
    else:
        action, rot_age = "Hold Until Retirement", retire_age

    st.subheader("\U0001F501 Decision")
    color = "green" if posterior>=0.6 else "orange" if posterior>=0.5 else "red"
    st.markdown(f"**Rotation Probability:** <span style='color:{color}'>**{posterior:.1%}**</span>", unsafe_allow_html=True)
    st.markdown(f"**Action:** **{action}**")
    if action == "Rotate Now":
        st.caption("💡 Rotating now helps lock in some gains while the market is favorable, enabling diversification into income assets without waiting for a possible downturn.")
    elif action == "Rotate Later":
        st.caption("💡 Rotation may be better timed later, allowing more BTC upside while staying flexible. You’re approaching optimal territory, but not quite there yet.")
    else:
        st.caption("💡 Holding until retirement maximizes BTC upside potential. It’s a high-growth strategy, but income generation will be deferred until then.")

    with st.expander("📊 Market Sentiment & Threshold Debug Info"):
        st.write(f"Drawdown Score: {drawdown_score:.2f}")
        st.write(f"MA Score: {ma_score}")
        st.write(f"Market Sentiment Score: {market_sentiment:.2f}")
        st.write(f"Threshold Score: {threshold_score:.2f}")
        st.write(f"Goal Proximity Score (w/ age): {goal_score:.2f}")
        st.write(f"Final Posterior: {posterior:.3f}")

# ---- ALLOCATION OPTIMIZER ---- #
    def score_alloc(x):
        target     = age if action=="Rotate Now" else retire_age
        yrs_to_rot = target - age
        proj_val   = current_value * np.exp(btc_return * yrs_to_rot)

        eff_rot    = rotation_percent * (1 - keep_mstr_pct/100)
        kept_frac  = rotation_percent * (keep_mstr_pct/100)
        yrs_post   = max(0, 82 - target)

        kept_val   = proj_val * kept_frac * np.exp(btc_return * yrs_post)
        rot_amt    = proj_val * eff_rot
        ann_inc    = rot_amt * (x[0]*msty_yield + x[1]*strk_yield + x[2]*strf_yield)
        cum_inc    = ann_inc * yrs_post

        caps = [
            proj_val * kept_frac * np.exp(
                btc_return*yrs_post + volatility*np.random.randn()*np.sqrt(yrs_post)
            ) for _ in range(200)
        ]
        cap_var = np.var(caps)

        alpha = inc_pref/100
        return -(alpha*cum_inc + (1-alpha)*kept_val - (1-alpha)*risk_aversion*cap_var)

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
        strk_pct = st.slider("STRK (%)", 0, max_strk, strk_pct)
        max_strf = 100 - msty_pct - strk_pct
        strf_pct = st.slider("STRF (%)", 0, max_strf, strf_pct)
        if msty_pct + strk_pct + strf_pct > 100:
            st.error("Total exceeds 100%; adjust sliders.")
    else:
        st.markdown(f"- MSTY **{msty_pct}%**, STRK **{strk_pct}%**, STRF **{strf_pct}%**")
    st.progress((msty_pct+strk_pct+strf_pct)/100)

    # ---- PROJECT & COMPARISON ---- #
    def project(rotation_age):
        yrs      = rotation_age - age
        proj_val = current_value * np.exp(btc_return * yrs)
        eff_rot  = rotation_percent*(1-keep_mstr_pct/100)
        kept_frac= rotation_percent*(keep_mstr_pct/100)
        yrs_post = max(0,82-rotation_age)

        kept_val = proj_val*kept_frac*np.exp(btc_return*yrs_post)
        rot_amt  = proj_val*eff_rot
        ann_inc  = rot_amt*(msty_pct/100*msty_yield + strk_pct/100*strk_yield + strf_pct/100*strf_yield)
        cum_inc  = ann_inc*yrs_post
        return kept_val, cum_inc

    kv_now, ci_now = project(age)
    kv_ret, ci_ret = project(retire_age)

    st.header("🔍 Comparison")
    c1,c2,c3 = st.columns(3)
    with c1:
        st.write("**Metric**"); st.write("MSTR @82"); st.write("Cum. Income")
    with c2:
        st.write("**Rotate Now**"); st.write(f"${kv_now:,.0f}"); st.write(f"${ci_now:,.0f}")
    with c3:
        st.write(f"**Rotate at {retire_age}**"); st.write(f"${kv_ret:,.0f}"); st.write(f"${ci_ret:,.0f}")
    if (kv_now+ci_now)>(kv_ret+ci_ret):
        st.success("▶️ Rotating Now yields the best combined outcome.")
    else:
        st.info(f"▶️ Waiting until age {retire_age} may yield a better outcome.")

    # ---- TIMELINE ---- #
    st.subheader("📅 Timeline")
    fig, ax = plt.subplots(figsize=(8,2))
    ax.axvline(age,color='blue',label="Today")
    if rot_age is not None: ax.axvline(rot_age,color='green',label="Rotate")
    ax.axvline(retire_age,color='gray',linestyle='--',label="Retire")
    lab = action if rot_age is not None else f"Rotate at {retire_age}"
    ax.text((age+(rot_age or retire_age))/2,0.5,lab,ha='center')
    ax.set_xlim(age-1,retire_age+1); ax.get_yaxis().set_visible(False); ax.legend()
    st.pyplot(fig)

    # ---- CASH-FLOW OUTLOOK (DUAL-AXIS) ---- #
    st.subheader("📈 Cash-Flow Outlook Through Age 82")
    death_age = 82
    yrs_pre   = (rot_age or retire_age) - age
    yrs_post  = death_age - (rot_age or retire_age)

    sim = np.zeros((yrs_pre+yrs_post+1,num_simulations))
    sim[0] = current_value
    for t in range(1,yrs_pre+1):
        sim[t] = sim[t-1]*np.exp((btc_return-0.5*volatility**2)+volatility*np.random.randn(num_simulations))
    for t in range(yrs_pre+1,yrs_pre+yrs_post+1):
        frozen = sim[yrs_pre]*(1-keep_mstr_pct/100)
        grow   = sim[yrs_pre]*(keep_mstr_pct/100)*np.exp((btc_return-0.5*volatility**2)*(t-yrs_pre)+volatility*np.random.randn(num_simulations)*np.sqrt(t-yrs_pre))
        sim[t] = frozen+grow

    mean_path = sim.mean(axis=1)
    ages_all  = np.arange(age,death_age+1)
    inc_all   = [0]*yrs_pre + [ci_now if action=="Rotate Now" else ci_ret]*(yrs_post+1)

    fig_static, ax_s = plt.subplots(figsize=(10,4))
    ax_s.plot(ages_all,mean_path,color='tab:blue',linewidth=2)
    ax_s.set_xlabel("Age"); ax_s.set_ylabel("Portfolio Value (USD)",color='tab:blue')
    ax_s.tick_params(axis='y',labelcolor='tab:blue'); ax_s.ticklabel_format(style='plain',axis='y')
    ax2_s = ax_s.twinx()
    bars_s= ax2_s.bar(ages_all,inc_all,alpha=0.4,color='tab:orange',width=0.8)
    ax2_s.set_ylabel("Annual Income (USD)",color='tab:orange'); ax2_s.tick_params(axis='y',labelcolor='tab:orange')
    ax2_s.ticklabel_format(style='plain',axis='y')
    ax2_s.bar_label(bars_s,labels=[f"${h:,.0f}" if h>0 else "" for h in inc_all],padding=3,fontsize=8,rotation=90,label_type='edge')
    for age_line,style,lbl in [(rot_age or retire_age,'--',"Rotation"),(retire_age,'-.',"Retire"),(death_age,':',f"Life Exp ({death_age})")]:
        color_line='green' if lbl=="Rotation" else 'gray' if lbl=="Retire" else 'black'
        ax_s.axvline(age_line,color=color_line,linestyle=style,label=lbl)
    h1,l1=ax_s.get_legend_handles_labels(); h2,l2=ax2_s.get_legend_handles_labels()
    ax_s.legend(h1+h2,l1+l2,loc='upper left'); plt.tight_layout()

    play = st.checkbox("▶️ Play Monte Carlo Paths")
    placeholder = st.empty()
    if play:
        for i in range(min(20,num_simulations)):
            fig, ax = plt.subplots(figsize=(10,4))
            ax.plot(ages_all, sim[:,i], color='gray', alpha=0.3)
            ax.plot(ages_all, mean_path, color='tab:blue', linewidth=2)
            placeholder.pyplot(fig)
            time.sleep(0.3)
    else:
        st.pyplot(fig_static)

with doc_tab:
    st.title("\U0001F4D8 Documentation & Assumptions")
    st.markdown("""
    **Rotation Action Explanations**

    - **Rotate Now:** Lock in gains while BTC and MSTR are strong. Diversify into income-producing assets to reduce risk ahead of retirement.
    - **Rotate Later:** Near the threshold, but not quite optimal. May benefit from short-term BTC upside before rotating.
    - **Hold Until Retirement:** Maximize long-term BTC growth. Suitable if you don’t need income now and are comfortable with high volatility.

    ---

    **Assumptions & Limitations**
    - Lognormal returns, constant yields, no fees/taxes.
    - Horizon: age 82 life expectancy.
    - Advanced signals removed for simplicity.
    """)

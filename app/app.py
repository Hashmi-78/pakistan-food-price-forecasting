import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Pakistan Food Price Early Warning System",
    page_icon="🌾",
    layout="wide"
)

# ── Load data ─────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('../data/processed/punjab_food_prices_features.csv', parse_dates=['date'])
    results = pd.read_csv('../outputs/model_comparison.csv')
    return df, results

@st.cache_resource
def load_models():
    wheat_model = joblib.load('../models/wheat_random_forest.pkl')
    rice_model = joblib.load('../models/rice_prophet.pkl')
    sugar_model = joblib.load('../models/sugar_prophet.pkl')
    return wheat_model, rice_model, sugar_model

df, results = load_data()
wheat_model, rice_model, sugar_model = load_models()

FEATURES = [c for c in df.columns if c not in
            ['date', 'wheat_price', 'rice_price', 'sugar_price', 'year']]

# ── Header ────────────────────────────────────────────────────
st.title("🌾 Pakistan Food Price Early Warning System")
st.markdown("**Punjab Region · 2013–2021 · Wheat Flour, Rice, Sugar**")
st.markdown("Predicting monthly food price movements to give farmers and buyers a 4-week early warning before a price shock hits.")
st.divider()

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.title("Controls")
commodity = st.sidebar.selectbox(
    "Select Commodity",
    ["Wheat Flour", "Rice (Basmati)", "Sugar"]
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Best model per commodity**")
st.sidebar.markdown("🌾 Wheat → Random Forest (MAE: 2.08 PKR/kg)")
st.sidebar.markdown("🍚 Rice → Prophet (MAE: 4.39 PKR/kg)")
st.sidebar.markdown("🍬 Sugar → Prophet (MAE: 3.23 PKR/kg)")

# ── Map commodity selection ───────────────────────────────────
col_map = {
    "Wheat Flour": "wheat_price",
    "Rice (Basmati)": "rice_price",
    "Sugar": "sugar_price"
}
color_map = {
    "Wheat Flour": "#2563EB",
    "Rice (Basmati)": "#16A34A",
    "Sugar": "#DC2626"
}
selected_col = col_map[commodity]
selected_color = color_map[commodity]

# ── Risk alert ────────────────────────────────────────────────
latest_price = df[selected_col].iloc[-1]
prev_price = df[selected_col].iloc[-2]
pct_change = ((latest_price - prev_price) / prev_price) * 100

st.subheader("📊 Current Risk Status")
col1, col2, col3 = st.columns(3)
col1.metric("Latest Price", f"PKR {latest_price:.2f}/kg", f"{pct_change:+.1f}% vs last month")
col2.metric("Dataset Period", "Jan 2013 – Jul 2021")
col3.metric("Total Records", f"{len(df)} months")

if pct_change > 5:
    st.error(f"🚨 HIGH RISK — {commodity} price rose {pct_change:.1f}% last month. Shock threshold exceeded.")
elif pct_change > 2:
    st.warning(f"⚠️ WATCH — {commodity} price rising. Monitor closely.")
else:
    st.success(f"✅ STABLE — {commodity} price movement within normal range.")

st.divider()

# ── Price history ─────────────────────────────────────────────
st.subheader(f"📈 {commodity} Price History")
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(df['date'], df[selected_col], color=selected_color, linewidth=2)

# Mark shocks
pct_changes = df[selected_col].pct_change() * 100
shocks = df[pct_changes > 5]
ax.scatter(shocks['date'], shocks[selected_col],
           color='red', zorder=5, s=60, label='Price shock (>5%)')
ax.set_ylabel('PKR/kg')
ax.set_xlabel('Date')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
st.pyplot(fig)
plt.close()

st.divider()

# ── Forecast ──────────────────────────────────────────────────
st.subheader(f"🔮 6-Month Forecast — {commodity}")

if commodity == "Wheat Flour":
    # Random Forest forecast — use last known row, step forward
    last_row = df[FEATURES].iloc[[-1]]
    forecast_prices = []
    current_row = df.iloc[-1].copy()

    for i in range(6):
        pred = wheat_model.predict(pd.DataFrame([current_row[FEATURES]]))[0]
        forecast_prices.append(pred)
        # Shift lags forward
        current_row['wheat_price_lag4'] = current_row['wheat_price_lag2']
        current_row['wheat_price_lag2'] = current_row['wheat_price_lag1']
        current_row['wheat_price_lag1'] = pred

    last_date = df['date'].iloc[-1]
    forecast_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=6, freq='MS')

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df['date'].iloc[-24:], df[selected_col].iloc[-24:],
            color=selected_color, linewidth=2, label='Historical')
    ax.plot(forecast_dates, forecast_prices,
            color='black', linewidth=2, linestyle='--', label='Forecast')
    ax.scatter(forecast_dates, forecast_prices, color='black', zorder=5, s=40)
    ax.axvline(x=last_date, color='gray', linestyle=':', linewidth=1)
    ax.set_ylabel('PKR/kg')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

else:
    # Prophet forecast
    prophet_model = rice_model if commodity == "Rice (Basmati)" else sugar_model
    prophet_input = df[['date', selected_col]].rename(
        columns={'date': 'ds', selected_col: 'y'}
    )
    future = prophet_model.make_future_dataframe(periods=6, freq='MS')
    forecast = prophet_model.predict(future)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df['date'].iloc[-24:], df[selected_col].iloc[-24:],
            color=selected_color, linewidth=2, label='Historical')
    forecast_tail = forecast.tail(6)
    ax.plot(forecast_tail['ds'], forecast_tail['yhat'],
            color='black', linewidth=2, linestyle='--', label='Forecast')
    ax.fill_between(forecast_tail['ds'],
                    forecast_tail['yhat_lower'],
                    forecast_tail['yhat_upper'],
                    alpha=0.2, color='gray', label='Confidence range')
    ax.axvline(x=df['date'].iloc[-1], color='gray', linestyle=':', linewidth=1)
    ax.set_ylabel('PKR/kg')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

st.divider()

# ── Model comparison ──────────────────────────────────────────
st.subheader("🤖 Model Performance Comparison")
pivot = results.pivot_table(
    index='Model', columns='Commodity', values='MAE (PKR/kg)'
).round(2)
st.dataframe(pivot, use_container_width=True)
st.caption("MAE = Mean Absolute Error in PKR/kg. Lower is better.")

st.divider()

# ── Seasonal risk calendar ────────────────────────────────────
st.subheader("📅 Seasonal Risk Calendar")
month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

df['month'] = df['date'].dt.month
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

for ax, (col, label, color) in zip(axes, [
    ('wheat_price', 'Wheat Flour', '#2563EB'),
    ('rice_price', 'Rice (Basmati)', '#16A34A'),
    ('sugar_price', 'Sugar', '#DC2626')
]):
    pct = df[col].pct_change() * 100
    df['_pct'] = pct
    monthly = df.groupby('month')['_pct'].mean()
    bar_colors = ['#DC2626' if v > 0 else '#16A34A' for v in monthly.values]
    ax.bar(monthly.index, monthly.values, color=bar_colors, alpha=0.85)
    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.set_title(f'{label}', fontweight='bold', fontsize=10)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_names, fontsize=7, rotation=45)
    ax.set_ylabel('Avg % Change')
    ax.grid(True, alpha=0.3, axis='y')

plt.suptitle('Average Price Change by Month (Red = rises, Green = falls)',
             fontsize=11, fontweight='bold')
plt.tight_layout()
st.pyplot(fig)
plt.close()

st.divider()

# ── Key insights ──────────────────────────────────────────────
st.subheader("💡 Key Project Insights")
st.markdown("""
- 🔴 **Sugar** is the most shock-prone commodity — 8 price shocks above 5% threshold (2013–2021)
- ⚡ **Wheat** had the most violent single shock — **+22.8% in June 2020** (COVID + near-zero rainfall compounded)
- 🟢 **Rice** is the most stable — only 1 shock in 8 years
- 🌧️ Rainfall below 20mm coincided with the worst wheat spike — drought stress amplified COVID disruption
- 📅 Highest risk months: **July** (wheat), **February** (rice), **August** (sugar)
- 🤖 Random Forest beats ML baselines for stable commodities; Prophet dominates for seasonal ones
""")

st.divider()
st.caption("Built by Muhammad Umar · Data: WFP VAM via HDX · Weather: Open-Meteo API · Model: Random Forest + Prophet")
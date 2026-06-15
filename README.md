# 🌾 Pakistan Food Price Early Warning System

> Predicting monthly food price movements of wheat flour, rice, and sugar across Punjab — giving farmers, buyers, and policymakers a **4-week early warning** before a price shock hits.

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20App-red)](https://streamlit.io/)
[![Data](https://img.shields.io/badge/Data-WFP%20VAM%20%7C%20HDX-orange)](https://data.humdata.org/dataset/wfp-food-prices-for-pakistan)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 🔴 The Problem

Small farmers in Punjab, Pakistan have no reliable way to know if food prices will spike next month before they decide what to plant, when to sell, or how much to store. The consequences of getting this wrong are severe — a 22% wheat price spike in a single month (June 2020) wiped out household budgets across the province.

This project builds a data-driven early warning system that flags price shock risk 4 weeks ahead, using historical price data, weather signals, and seasonal patterns.

---

## 🎯 What This Project Does

- Forecasts next-month prices for **wheat flour, rice (basmati), and sugar** in Punjab
- Flags **price shock risk** (>5% month-on-month rise) with colour-coded alerts
- Compares **4 modelling approaches**: Linear Regression, Random Forest, XGBoost, and Facebook Prophet
- Identifies **seasonal risk windows** — the specific months historically most dangerous for each commodity
- Deep-dives into the **2020 wheat crisis** — the most severe price shock in the dataset

---

## 📊 Key Findings

| Finding | Detail |
|---------|--------|
| Most shock-prone commodity | **Sugar** — 8 price shocks above 5% threshold (2013–2021) |
| Most violent single shock | **Wheat +22.8% in June 2020** — COVID-19 + near-zero rainfall compounded |
| Most stable commodity | **Rice** — only 1 shock in 8 years |
| Highest risk month — Wheat | **July** — post-harvest stock depletion |
| Highest risk month — Rice | **February** — seasonal supply tightening |
| Highest risk month — Sugar | **August** — post-crushing season demand peak |
| Weather insight | Rainfall below 20mm coincided with the worst wheat spike — drought amplified COVID disruption |

---

## 🤖 Model Performance

| Model | Wheat MAE (PKR/kg) | Rice MAE (PKR/kg) | Sugar MAE (PKR/kg) |
|-------|-------------------|------------------|-------------------|
| Linear Regression | 4.72 | 4.88 | 4.38 |
| Random Forest | **2.08** ✅ | 5.18 | 7.01 |
| XGBoost | 2.17 | 5.45 | 7.10 |
| Prophet | 4.14 | **4.39** ✅ | **3.23** ✅ |

**Key insight:** Random Forest dominates for wheat (stable, trend-driven commodity). Prophet dominates for rice and sugar (seasonality-heavy, policy-driven commodities). Tree-based models underperform on sugar because price shocks there are driven by sudden policy decisions — not historical patterns.

---

## 🗂️ Project Structure

```
pakistan-food-price-forecasting/
│
├── data/
│   ├── raw/                              # Original downloaded files — never edited
│   │   └── wfp_food_prices_pakistan.csv
│   └── processed/                        # Cleaned and feature-engineered files
│       ├── punjab_food_prices_clean.csv
│       ├── punjab_food_prices_merged.csv
│       └── punjab_food_prices_features.csv
│
├── notebooks/
│   ├── phase1_data_collection.ipynb      # Data loading, filtering, cleaning
│   ├── phase2_eda_feature_engineering.ipynb  # EDA, correlation, feature engineering
│   ├── phase3_modelling.ipynb            # All 4 models, evaluation, comparison
│   └── phase4_insights.ipynb            # Price shocks, seasonal calendar, COVID deep dive
│
├── outputs/
│   ├── price_trends.png                  # Historical price charts
│   ├── seasonality_by_month.png          # Avg price by month
│   ├── rainfall_correlation.png          # Weather vs price scatter
│   ├── correlation_matrix.png            # Full correlation heatmap
│   ├── price_shocks.png                  # Shock detection chart
│   ├── seasonal_risk_calendar.png        # Monthly risk calendar
│   ├── covid_wheat_shock.png             # 2020 crisis deep dive
│   ├── actual_vs_predicted.png           # Model predictions vs actuals
│   ├── model_comparison_chart.png        # MAE comparison bar chart
│   ├── model_comparison.csv              # Full results table
│   └── project_insights.csv             # Summary of all findings
│
├── models/
│   ├── wheat_random_forest.pkl           # Best model for wheat
│   ├── rice_prophet.pkl                  # Best model for rice
│   └── sugar_prophet.pkl                 # Best model for sugar
│
├── app/
│   └── app.py                            # Streamlit app
│
├── requirements.txt
└── README.md
```

---

## 📦 Data Sources

| Source | Data | Coverage |
|--------|------|----------|
| [WFP VAM via HDX](https://data.humdata.org/dataset/wfp-food-prices-for-pakistan) | Food prices (PKR/kg) by commodity and city | 2004–2021 |
| [Open-Meteo Historical API](https://open-meteo.com/en/docs/historical-weather-api) | Daily rainfall (mm) and temperature (°C) for Lahore | 2013–2021 |

---

## 🧠 Feature Engineering

| Feature Type | Features | Rationale |
|-------------|----------|-----------|
| Lag features | Price 1, 2, 4 months ago | Past prices are the strongest predictor of future prices |
| Rolling averages | 4-month and 8-month moving averages | Captures trend direction |
| Price momentum | % change over last 2 months | Captures acceleration |
| Weather lags | Rainfall 2 and 4 months ago | Crop growth delay — weather affects supply weeks later |
| Season flags | is_summer, is_harvest, is_crushing | Encodes agricultural calendar as domain knowledge |

---

## 🚀 Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/yourusername/pakistan-food-price-forecasting.git
cd pakistan-food-price-forecasting
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Download the data**

Download the WFP Pakistan food prices CSV from [HDX](https://data.humdata.org/dataset/wfp-food-prices-for-pakistan) and place it in `data/raw/wfp_food_prices_pakistan.csv`

**4. Run the notebooks in order**
```
notebooks/phase1_data_collection.ipynb
notebooks/phase2_eda_feature_engineering.ipynb
notebooks/phase3_modelling.ipynb
notebooks/phase4_insights.ipynb
```

**5. Launch the Streamlit app**
```bash
cd app
streamlit run app.py
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.12 | Core language |
| Pandas | Data manipulation |
| Matplotlib / Seaborn | Visualisation |
| Scikit-learn | Linear Regression, Random Forest |
| XGBoost | Gradient boosting |
| Facebook Prophet | Time series forecasting |
| Streamlit | Interactive web app |
| Joblib | Model serialisation |
| Open-Meteo API | Weather data |

---

## ⚠️ Limitations

- **Dataset ends July 2021** — does not capture post-2021 inflation surge or 2022–23 economic crisis
- **Weather data is Lahore-only** — Punjab is large; regional weather variation is not captured
- **Monthly granularity** — weekly or daily data would improve short-term shock detection
- **No policy variables** — government interventions (price floors, export bans, subsidies) are not modelled. The 2019–2020 sugar crisis was largely policy-driven, which explains why ML models underperform on sugar vs Prophet
- **103 rows of training data** — a larger dataset would meaningfully improve model reliability

---

## 💡 Future Improvements

- Add weekly price data for finer-grained forecasting
- Include fuel/diesel prices as a feature (transport cost proxy)
- Incorporate Google Trends data for demand-side signals
- Extend to tomato and onion — the most emotionally volatile commodities in Pakistan
- Deploy with auto-updating data pipeline via scheduled API calls

---

## 👤 About

Built by **[Your Name]** as a portfolio project during final year of BSc Data Science.

The motivation is personal — food price volatility in Pakistan directly affects millions of households. This project attempts to give data-driven visibility to a problem that is usually only noticed after a crisis has already hit.

📧 [your.email@gmail.com] · 🔗 [LinkedIn] · 💻 [GitHub]

---

## 📄 License

MIT License — free to use, modify, and distribute with attribution.
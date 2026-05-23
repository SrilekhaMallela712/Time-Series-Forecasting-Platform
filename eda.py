import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller

# =========================
# LOAD DATASET
# =========================

df = pd.read_csv("data/train.csv")

# =========================
# BASIC INFORMATION
# =========================

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# =========================
# DATE CONVERSION
# =========================

df['date'] = pd.to_datetime(df['date'])

# =========================
# DAILY SALES AGGREGATION
# =========================

daily_sales = (
    df.groupby('date')['sales']
    .sum()
    .reset_index()
)

print("\nDaily Sales:")
print(daily_sales.head())

# =========================
# ROLLING MEAN
# =========================

daily_sales['rolling_mean_30'] = (
    daily_sales['sales']
    .rolling(window=30)
    .mean()
)

# =========================
# INTERACTIVE PLOT
# =========================

fig = px.line(
    daily_sales,
    x='date',
    y=['sales', 'rolling_mean_30'],
    title='Daily Sales Trend',
)

fig.show()

# =========================
# STL DECOMPOSITION
# =========================

stl = STL(daily_sales['sales'], period=7)

result = stl.fit()

result.plot()

plt.show()

# =========================
# ADF TEST
# =========================

adf_result = adfuller(daily_sales['sales'])

print("\nADF Statistic:", adf_result[0])
print("p-value:", adf_result[1])

if adf_result[1] < 0.05:
    print("Time Series is Stationary")
else:
    print("Time Series is Non-Stationary")
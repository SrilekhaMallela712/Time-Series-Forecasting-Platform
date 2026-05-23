import pandas as pd
import numpy as np

# =========================
# LOAD DATASET
# =========================

df = pd.read_csv("data/train.csv")

# =========================
# DATE CONVERSION
# =========================

df['date'] = pd.to_datetime(df['date'])

# =========================
# SORT DATA
# =========================

df = df.sort_values('date')

# =========================
# CREATE LAG FEATURES
# =========================

df['lag_1'] = df['sales'].shift(1)

df['lag_7'] = df['sales'].shift(7)

df['lag_30'] = df['sales'].shift(30)

# =========================
# ROLLING FEATURES
# =========================

df['rolling_mean_7'] = (
    df['sales']
    .rolling(window=7)
    .mean()
)

df['rolling_std_7'] = (
    df['sales']
    .rolling(window=7)
    .std()
)

# =========================
# DATE FEATURES
# =========================

df['day_of_week'] = df['date'].dt.dayofweek

df['month'] = df['date'].dt.month

df['year'] = df['date'].dt.year

df['week_of_year'] = df['date'].dt.isocalendar().week

# =========================
# REMOVE NULL VALUES
# =========================

df = df.dropna()

# =========================
# FINAL OUTPUT
# =========================

print("\nProcessed Dataset:")

print(df.head())

print("\nProcessed Shape:")

print(df.shape)

# =========================
# SAVE PROCESSED DATA
# =========================

df.to_csv("exports/processed_data.csv", index=False)

print("\nProcessed data saved successfully.")
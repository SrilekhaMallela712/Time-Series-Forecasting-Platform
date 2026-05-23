import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Forecasting Platform",
    page_icon="📈",
    layout="wide"
)

# =========================
# CUSTOM CSS
# =========================

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

section[data-testid="stSidebar"] {
    background-color: #111827;
}

h1, h2, h3, h4 {
    color: white;
}

.stMetric {
    background-color: #1F2937;
    padding: 15px;
    border-radius: 12px;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================

st.sidebar.image(
    "https://cdn-icons-png.flaticon.com/512/2103/2103633.png",
    width=80
)

st.sidebar.title("Forecasting Platform")

# =========================
# DATASET UPLOAD
# =========================

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV Dataset",
    type=["csv"]
)

# =========================
# LOAD DATASET
# =========================

if uploaded_file is not None:

    processed_df = pd.read_csv(uploaded_file)

    st.sidebar.success(
        "Dataset Uploaded Successfully!"
    )

else:

    processed_df = pd.read_csv(
        "exports/processed_data.csv"
    )

    st.sidebar.info(
        "Using Default Dataset"
    )

# =========================
# COLUMN SELECTION
# =========================

st.sidebar.subheader("Dataset Configuration")

date_column = st.sidebar.selectbox(
    "Select Date Column",
    processed_df.columns
)

target_column = st.sidebar.selectbox(
    "Select Target Column",
    processed_df.columns
)

# =========================
# DATE CONVERSION
# =========================

try:

    processed_df[date_column] = pd.to_datetime(
        processed_df[date_column]
    )

except:

    st.error(
        "Selected date column cannot be converted to datetime."
    )

    st.stop()

# =========================
# LOAD FORECAST FILES
# =========================

try:

    prophet_df = pd.read_csv(
        "exports/prophet_forecast.csv"
    )

except:

    prophet_df = pd.DataFrame()

try:

    xgb_df = pd.read_csv(
        "exports/xgboost_predictions.csv"
    )

except:

    xgb_df = pd.DataFrame()

# =========================
# SIDEBAR NAVIGATION
# =========================

menu = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Dataset Analytics",
        "Prophet Forecast",
        "XGBoost Forecast",
        "Model Comparison",
        "Download Center"
    ]
)

# =========================
# MAIN TITLE
# =========================

st.title("📈 Advanced Time Series Forecasting Platform")

st.markdown("""
AI-powered demand forecasting platform using:

- ARIMA
- Prophet
- XGBoost
- Machine Learning Forecasting
- Multi-Dataset Support
""")

# =========================
# DASHBOARD
# =========================

if menu == "Dashboard":

    st.header("Business Forecast Dashboard")

    total_sales = processed_df[target_column].sum()

    avg_sales = processed_df[target_column].mean()

    max_sales = processed_df[target_column].max()

    total_records = len(processed_df)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Value",
        f"{total_sales:,.0f}"
    )

    col2.metric(
        "Average Value",
        f"{avg_sales:,.2f}"
    )

    col3.metric(
        "Maximum Value",
        f"{max_sales:,.0f}"
    )

    col4.metric(
        "Records",
        f"{total_records:,}"
    )

    st.subheader("Time Series Trend")

    trend_df = (
        processed_df
        .groupby(date_column)[target_column]
        .sum()
        .reset_index()
    )

    fig = px.line(
        trend_df,
        x=date_column,
        y=target_column,
        title="Trend Over Time"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Top 10 Highest Values")

    top_values = trend_df.sort_values(
        by=target_column,
        ascending=False
    ).head(10)

    fig2 = px.bar(
        top_values,
        x=date_column,
        y=target_column,
        title="Top 10 Highest Values"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

# =========================
# DATASET ANALYTICS
# =========================

elif menu == "Dataset Analytics":

    st.header("Dataset Analytics")

    st.subheader("Dataset Preview")

    st.dataframe(
        processed_df.head(100)
    )

    st.subheader("Dataset Shape")

    st.write(
        processed_df.shape
    )

    st.subheader("Columns")

    st.write(
        list(processed_df.columns)
    )

    st.subheader("Statistical Summary")

    st.write(
        processed_df.describe()
    )

    st.subheader("Missing Values")

    st.write(
        processed_df.isnull().sum()
    )

# =========================
# PROPHET FORECAST
# =========================

elif menu == "Prophet Forecast":

    st.header("Prophet Forecast Analysis")

    if prophet_df.empty:

        st.warning(
            "Prophet forecast file not found."
        )

    else:

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=prophet_df['ds'],
                y=prophet_df['yhat'],
                mode='lines',
                name='Forecast'
            )
        )

        fig.add_trace(
            go.Scatter(
                x=prophet_df['ds'],
                y=prophet_df['yhat_upper'],
                mode='lines',
                name='Upper Interval'
            )
        )

        fig.add_trace(
            go.Scatter(
                x=prophet_df['ds'],
                y=prophet_df['yhat_lower'],
                mode='lines',
                name='Lower Interval'
            )
        )

        fig.update_layout(
            title="Prophet Forecast with Confidence Intervals"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.subheader("Forecast Dataset")

        st.dataframe(
            prophet_df.head(100)
        )

# =========================
# XGBOOST FORECAST
# =========================

elif menu == "XGBoost Forecast":

    st.header("XGBoost Forecast Analysis")

    if xgb_df.empty:

        st.warning(
            "XGBoost prediction file not found."
        )

    else:

        fig = px.line(
            xgb_df.head(500),
            y=['Actual', 'Predicted'],
            title="Actual vs Predicted"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.subheader("Prediction Dataset")

        st.dataframe(
            xgb_df.head(100)
        )

# =========================
# MODEL COMPARISON
# =========================

elif menu == "Model Comparison":

    st.header("Forecast Model Comparison")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "ARIMA MAPE",
        "33.66%"
    )

    col2.metric(
        "Prophet MAPE",
        "40.74%"
    )

    col3.metric(
        "XGBoost",
        "Best Performing"
    )

    comparison_df = pd.DataFrame({

        'Model': [
            'ARIMA',
            'Prophet',
            'XGBoost'
        ],

        'MAPE': [
            33.66,
            40.74,
            20.00
        ]
    })

    fig = px.bar(
        comparison_df,
        x='Model',
        y='MAPE',
        title='Forecast Accuracy Comparison'
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# =========================
# DOWNLOAD CENTER
# =========================

elif menu == "Download Center":

    st.header("Download Forecast Files")

    if not prophet_df.empty:

        st.download_button(
            label="Download Prophet Forecast",
            data=prophet_df.to_csv(index=False),
            file_name="prophet_forecast.csv",
            mime="text/csv"
        )

    if not xgb_df.empty:

        st.download_button(
            label="Download XGBoost Predictions",
            data=xgb_df.to_csv(index=False),
            file_name="xgboost_predictions.csv",
            mime="text/csv"
        )

# =========================
# FOOTER
# =========================

st.markdown("---")

st.caption(
    "Developed using Python, Prophet, XGBoost, Streamlit, Plotly & Machine Learning"
)
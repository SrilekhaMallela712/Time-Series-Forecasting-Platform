import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# ==========================
# PAGE CONFIG
# ==========================

st.set_page_config(
    page_title="Advanced Forecasting Platform",
    page_icon="📈",
    layout="wide"
)

# ==========================
# CUSTOM CSS
# ==========================

st.markdown("""
<style>

.main{
background-color:#0E1117;
}

h1,h2,h3{
color:white;
}

</style>
""", unsafe_allow_html=True)

# ==========================
# SIDEBAR
# ==========================

st.sidebar.image(
    "https://cdn-icons-png.flaticon.com/512/2103/2103633.png",
    width=80
)

st.sidebar.title(
    "Forecasting Platform"
)

# ==========================
# FILE UPLOAD
# ==========================

uploaded_file=st.sidebar.file_uploader(
    "Upload CSV Dataset",
    type=["csv"]
)

if uploaded_file:

    processed_df=pd.read_csv(
        uploaded_file
    )

    st.sidebar.success(
        "Dataset uploaded successfully"
    )

else:

    try:

        processed_df=pd.read_csv(
            "exports/processed_data.csv"
        )

        st.sidebar.success(
            "Using default dataset"
        )

    except:

        st.warning(
            "Upload dataset to continue"
        )

        st.stop()

# ==========================
# COLUMN SELECTION
# ==========================

st.sidebar.subheader(
    "Dataset Configuration"
)

date_col=st.sidebar.selectbox(
    "Select Date Column",
    processed_df.columns
)

target_col=st.sidebar.selectbox(
    "Select Target Column",
    processed_df.select_dtypes(
        include=['int64','float64']
    ).columns
)

# ==========================
# DATE CONVERSION
# ==========================

try:

    processed_df[date_col]=pd.to_datetime(
        processed_df[date_col]
    )

except:

    st.error(
        "Date column conversion failed"
    )

    st.stop()

# ==========================
# LOAD FORECAST FILES
# ==========================

try:

    prophet_df=pd.read_csv(
        "exports/prophet_forecast.csv"
    )

except:

    prophet_df=pd.DataFrame()

try:

    xgb_df=pd.read_csv(
        "exports/xgboost_predictions.csv"
    )

except:

    xgb_df=pd.DataFrame()

# ==========================
# NAVIGATION
# ==========================

menu=st.sidebar.radio(

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

# ==========================
# TITLE
# ==========================

st.title(
"📈 Advanced Time Series Forecasting Platform"
)

st.markdown("""

AI-powered forecasting system using:

- ARIMA
- Prophet
- XGBoost
- Machine Learning

""")

# ==========================
# DASHBOARD
# ==========================

if menu=="Dashboard":

    st.header(
    "Business Dashboard"
    )

    total=processed_df[target_col].sum()

    avg=processed_df[target_col].mean()

    max_val=processed_df[target_col].max()

    total_records=len(
    processed_df
    )

    col1,col2,col3,col4=st.columns(
    4
    )

    col1.metric(
    "Total",
    f"{total:,.0f}"
    )

    col2.metric(
    "Average",
    f"{avg:,.2f}"
    )

    col3.metric(
    "Maximum",
    f"{max_val:,.0f}"
    )

    col4.metric(
    "Records",
    total_records
    )

    trend_df=(

    processed_df
    .groupby(
    date_col
    )[target_col]
    .sum()
    .reset_index()

    )

    fig=px.line(

    trend_df,

    x=date_col,
    y=target_col,
    title="Trend Over Time"

    )

    st.plotly_chart(
    fig,
    use_container_width=True
    )

# ==========================
# DATA ANALYTICS
# ==========================

elif menu=="Dataset Analytics":

    st.header(
    "Dataset Analytics"
    )

    st.subheader(
    "Dataset Preview"
    )

    st.dataframe(
    processed_df.head(
    100
    )
    )

    st.subheader(
    "Shape"
    )

    st.write(
    processed_df.shape
    )

    st.subheader(
    "Statistics"
    )

    st.write(
    processed_df.describe()
    )

# ==========================
# PROPHET
# ==========================

elif menu=="Prophet Forecast":

    st.header(
    "Prophet Forecast"
    )

    if len(
    prophet_df
    )>0:

        fig=go.Figure()

        fig.add_trace(

        go.Scatter(

        x=prophet_df["ds"],
        y=prophet_df["yhat"],
        mode='lines',
        name="Forecast"

        )

        )

        fig.add_trace(

        go.Scatter(

        x=prophet_df["ds"],
        y=prophet_df["yhat_upper"],
        mode='lines',
        name="Upper"

        )

        )

        fig.add_trace(

        go.Scatter(

        x=prophet_df["ds"],
        y=prophet_df["yhat_lower"],
        mode='lines',
        name="Lower"

        )

        )

        st.plotly_chart(
        fig,
        use_container_width=True
        )

    else:

        st.warning(
        "No Prophet forecast file found"
        )

# ==========================
# XGBOOST
# ==========================

elif menu=="XGBoost Forecast":

    st.header(
    "XGBoost Forecast"
    )

    if len(
    xgb_df
    )>0:

        fig=px.line(

        xgb_df.head(
        500
        ),

        y=[
        'Actual',
        'Predicted'
        ],

        title="Actual vs Predicted"

        )

        st.plotly_chart(
        fig,
        use_container_width=True
        )

    else:

        st.warning(
        "No prediction file found"
        )

# ==========================
# MODEL COMPARISON
# ==========================

elif menu=="Model Comparison":

    st.header(
    "Model Comparison"
    )

    compare=pd.DataFrame({

    "Model":[

    "ARIMA",
    "Prophet",
    "XGBoost"

    ],

    "MAPE":[

    33.66,
    40.74,
    20

    ]

    })

    fig=px.bar(

    compare,

    x="Model",
    y="MAPE",
    title="Forecast Accuracy"

    )

    st.plotly_chart(
    fig,
    use_container_width=True
    )

# ==========================
# DOWNLOAD
# ==========================

elif menu=="Download Center":

    st.header(
    "Download Forecast Files"
    )

    if len(prophet_df)>0:

        st.download_button(

        "Download Prophet",

        prophet_df.to_csv(
        index=False
        ),

        "prophet_forecast.csv",

        "text/csv"

        )

    if len(xgb_df)>0:

        st.download_button(

        "Download XGBoost",

        xgb_df.to_csv(
        index=False
        ),

        "xgboost_predictions.csv",

        "text/csv"

        )

# ==========================
# FOOTER
# ==========================

st.markdown("---")

st.caption(
"Developed using Python | Streamlit | Plotly | Prophet | XGBoost"
)

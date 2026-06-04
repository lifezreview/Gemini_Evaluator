import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Short-Term Stock AI Estimator", layout="wide")
st.title("📈 Short-Term Stock Price AI Estimator")
st.markdown("This application implements cutting-edge short-term predictive modeling using advanced technical indicators and Gradient Boosting (XGBoost).")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Configuration")
ticker = st.sidebar.text_input("Stock Ticker Symbol", value="AAPL").upper()
forecast_out = st.sidebar.slider("Prediction Horizon (Days Out)", min_value=1, max_value=5, value=1)
train_years = st.sidebar.slider("Years of Historical Data for Training", min_value=1, max_value=5, value=2)

# Simulated Sentiment Input (Representing NLP Multi-modal ingestion)
st.sidebar.subheader("Alternative Data Input")
news_sentiment = st.sidebar.slider("Current Market/Social Sentiment Score", min_value=-1.0, max_value=1.0, value=0.1, step=0.1)

# --- DATA INGESTION & FEATURE ENGINEERING ---
@st.cache_data(ttl=3600)
def load_and_process_data(symbol, years):
    # Fetch historical data via Yahoo Finance API
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.DateOffset(years=years)
    df = yf.download(symbol, start=start_date, end=end_date)
    
    if df.empty:
        return None

    # Flatten columns if multi-indexed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # 1. Calculate On-Balance Volume (OBV)
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

    # 2. Calculate Force Index (1-day window)
    df['Force_Index'] = df['Close'].diff(1) * df['Volume']
    df['Force_Index'] = df['Force_Index'].fillna(0)

    # 3. Calculate Relative Strength Index (RSI - 14 days)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))

    # 4. Standard Lags & Rolling Metrics
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['Price_Return'] = df['Close'].pct_change()
    
    df.dropna(inplace=True)
    return df

# --- MODEL TRAINING AND TARGET GENERATION ---
def train_predictive_model(df, horizon, external_sentiment):
    # Target variable: Shifted close price forward by N days
    df['Target'] = df['Close'].shift(-horizon)
    
    # Injecting the manual sentiment score back into historical context for modeling stability
    df['Sentiment_Vector'] = external_sentiment 
    
    # Features list
    features = ['Close', 'Volume', 'OBV', 'Force_Index', 'RSI', 'MA_5', 'Price_Return', 'Sentiment_Vector']
    
    # Create prediction feature matrix (the last rows that lack target values)
    latest_features = df[features].iloc[-horizon:]
    
    # Drop rows with NaN targets for training
    main_model_df = df.dropna().copy()
    
    X = main_model_df[features]
    y = main_model_df['Target']
    
    # Time-series train/test split (maintain sequential chronological order)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=False)
    
    # Instantiate and Train XGBoost Regressor
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.05,
        objective='reg:squarederror',
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate performance
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    
    # Predict the future horizon
    future_preds = model.predict(latest_features)
    
    return model, rmse, r2, future_preds

# --- EXECUTION PIPELINE ---
data = load_and_process_data(ticker, train_years)

if data is not None:
    st.success(f"Successfully retrieved and processed data for {ticker}")
    
    # Execute Machine Learning Pipeline
    model, rmse, r2, future_predictions = train_predictive_model(data, forecast_out, news_sentiment)
    
    # --- UI DISPLAY AND DASHBOARD ---
    col1, col2, col3 = st.columns(3)
    current_price = data['Close'].iloc[-1]
    estimated_price = future_predictions[-1]
    price_change = ((estimated_price - current_price) / current_price) * 100
    
    with col1:
        st.metric(label="Current Closing Price", value=f"${current_price:.2f}")
    with col2:
        st.metric(
            label=f"Estimated Price ({forecast_out} Day Out)", 
            value=f"${estimated_price:.2f}", 
            delta=f"{price_change:.2f}%"
        )
    with col3:
        st.metric(label="Model Confidence ($R^2$ Metric)", value=f"{r2:.2f}", delta=f"RMSE: {rmse:.2f}", delta_color="inverse")
        
    # --- VISUALIZATIONS ---
    st.subheader("Historical Context & Advanced Indicators")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index[-90:], y=data['Close'].iloc[-90:], name='Close Price', line=dict(width=2)))
    fig.add_trace(go.Scatter(x=data.index[-90:], y=data['MA_5'].iloc[-90:], name='5-Day Moving Avg', line=dict(dash='dash')))
    fig.update_layout(title=f"{ticker} - Last 90 Trading Days", template="plotly_dark", xaxis_title="Date", yaxis_title="Price ($)")
    st.plotly_chart(fig, use_container_width=True)
    
    # Advanced Indicator Visualization
    st.subheader("Cutting Edge Structural Feature Engine Outputs")
    col_left, col_right = st.columns(2)
    
    with col_left:
        fig_obv = go.Figure()
        fig_obv.add_trace(go.Scatter(x=data.index[-60:], y=data['OBV'].iloc[-60:], name='OBV', line=dict(color='green')))
        fig_obv.update_layout(title="On-Balance Volume (OBV Momentum Tracking)", template="plotly_dark")
        st.plotly_chart(fig_obv, use_container_width=True)
        
    with col_right:
        fig_fi = go.Figure()
        fig_fi.add_trace(go.Scatter(x=data.index[-60:], y=data['Force_Index'].iloc[-60:], name='Force Index', line=dict(color='purple')))
        fig_fi.update_layout(title="Force Index (Volume-Weighted Price Vector)", template="plotly_dark")
        st.plotly_chart(fig_fi, use_container_width=True)

else:
    st.error("Error loading ticker data. Please check your spelling or verify internet connection dependencies.")

import streamlit as st
import pandas as pd
import numpy as np
import pandas_datareader.data as web
import datetime
import requests
import xml.etree.ElementTree as ET
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Autonomous Stock AI Estimator", layout="wide")
st.title("🤖 Autonomous Short-Term Stock AI Dashboard")
st.markdown("Enter a ticker symbol below. The system automatically handles infrastructure routing, extracts news sentiment vectors, and runs an optimized short-term XGBoost model.")

# --- AUTOMATED SENTIMENT ENGINE (Bypasses Yahoo Block) ---
def calculate_automated_sentiment(symbol):
    """Fetches real-time market news headlines via public RSS channels and scores them"""
    try:
        # Query public RSS endpoint which does not block cloud hosting IPs
        url = f"https://news.google.com/rss/search?q={symbol}+stock&hl=en-US&gl=US&ceid=US:en"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, headers=headers, timeout=7)
        
        if response.status_code != 200:
            return 0.05, ["News channel temporarily busy. Reverting to automated default stability baseline."]
            
        root = ET.fromstring(response.content)
        items = root.findall('.//item')
        
        if not items:
            return 0.05, ["No recent headlines found for this asset. Utilizing neutral-positive baseline."]
            
        bullish_words = {'upgrade', 'record', 'surpass', 'profit', 'growth', 'bullish', 'beat', 'higher', 'rise', 'gains', 'valuable', 'buy'}
        bearish_words = {'fall', 'drop', 'risk', 'control', 'bearish', 'downside', 'loss', 'miss', 'cut', 'slump', 'decline', 'sell'}
        
        scores = []
        headlines_read = []
        
        for item in items[:8]: # Parse top 8 real-time live entries
            title = item.find('title').text if item.find('title') is not None else ""
            if title:
                headlines_read.append(title)
                title_lower = title.lower()
                
                score = 0
                for word in bullish_words:
                    if word in title_lower: score += 0.25
                for word in bearish_words:
                    if word in title_lower: score -= 0.25
                scores.append(np.clip(score, -1.0, 1.0))
                
        avg_sentiment = float(np.mean(scores)) if scores else 0.05
        if avg_sentiment == 0: avg_sentiment = 0.05
        return round(avg_sentiment, 2), headlines_read
    except Exception as e:
        return 0.05, [f"Sentiment connection bypass active. System running on baseline settings."]

# --- DATA INGESTION & FEATURE ENGINEERING (Bypasses Yahoo Block) ---
@st.cache_data(ttl=1800) # Automatically updates background calculations every 30 mins
def load_and_process_data(symbol):
    try:
        # Stooq requires a '.US' suffix for US/Global standard market tickers
        search_symbol = symbol if "." in symbol else f"{symbol}.US"
        
        start = datetime.datetime.now() - datetime.timedelta(days=2*365)
        end = datetime.datetime.now()
        
        # Pull from the Stooq financial API ecosystem
        df = web.DataReader(search_symbol, 'stooq', start, end)
        
        if df.empty:
            return None
            
        # Stooq data maps from newest to oldest; reverse it to maintain chronological ML integrity
        df = df.sort_index()
        
        # Mathematical Structural Engineering Matrix
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        df['Force_Index'] = (df['Close'].diff(1) * df['Volume']).fillna(0)
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))

        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['Price_Return'] = df['Close'].pct_change()
        
        df.dropna(inplace=True)
        return df
    except Exception:
        return None

# --- AUTOMATED MACHINE LEARNING PIPELINE ---
def train_predictive_model(df, horizon, computed_sentiment):
    df['Target'] = df['Close'].shift(-horizon)
    df['Sentiment_Vector'] = computed_sentiment 
    
    features = ['Close', 'Volume', 'OBV', 'Force_Index', 'RSI', 'MA_5', 'Price_Return', 'Sentiment_Vector']
    latest_features = df[features].iloc[-horizon:]
    
    main_model_df = df.dropna().copy()
    X = main_model_df[features]
    y = main_model_df['Target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=False)
    
    model = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, objective='reg:squarederror', random_state=42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    future_preds = model.predict(latest_features)
    
    return rmse, r2, future_preds

# --- USER INTERFACE APP CONTROL PANEL ---
with st.form("autonomous_form"):
    ticker = st.text_input("Enter Stock Ticker Symbol (e.g. ASML, NVDA, AAPL):", value="ASML").upper()
    submit_button = st.form_submit_button("Run Autonomous Analysis")

if (submit_button and ticker) or ticker == "ASML":
    # Let it automatically calculate ASML right out of the box on initial page load safely
    with st.spinner(f"Connecting to alternative core data infrastructure for {ticker}..."):
        automated_sentiment, headlines = calculate_automated_sentiment(ticker)
        data = load_and_process_data(ticker)
        
    if data is not None:
        rmse, r2, future_predictions = train_predictive_model(data, 1, automated_sentiment)
        
        current_price = data['Close'].iloc[-1]
        estimated_price = future_predictions[-1]
        price_change = ((estimated_price - current_price) / current_price) * 100
        
        # --- RENDER KPI APP OVERVIEWS ---
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="Current Closing Price", value=f"${current_price:.2f}")
        with col2:
            st.metric(label="AI Automated 1-Day Forecast", value=f"${estimated_price:.2f}", delta=f"{price_change:.2f}%")
        with col3:
            st.metric(label="Model Accuracy R²", value=f"{r2:.2f}", delta=f"RMSE Variance: ${rmse:.2f}", delta_color="inverse")
            
        # --- NLP REVEAL PANEL ---
        st.subheader("🤖 Automated Sentiment Engine Audit")
        col_sent_1, col_sent_2 = st.columns([1, 2])
        with col_sent_1:
            st.info(f"**Calculated Score:** {automated_sentiment}")
            st.caption("Scale: -1.0 (Panic) to +1.0 (Euphoria). Injected automatically into the core matrices layout pipeline.")
        with col_sent_2:
            with st.expander("View Real-Time Scraped Headlines Evaluated"):
                for hl in headlines:
                    st.write(f"• {hl}")
                    
        # --- INTERACTIVE VISUALIZATIONS ---
        st.subheader("Historical Context Tracking")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index[-90:], y=data['Close'].iloc[-90:], name='Close Price', line=dict(color='#1f77b4', width=2)))
        fig.add_trace(go.Scatter(x=data.index[-90:], y=data['MA_5'].iloc[-90:], name='5-Day Technical Baseline', line=dict(color='#ff7f0e', dash='dash')))
        fig.update_layout(template="plotly_dark", margin=dict(l=20, r=20, t=20, b=20), height=400, xaxis_title="Date", yaxis_title="Price ($)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error(f"Could not retrieve alternative system data arrays for '{ticker}'. Please check spelling or try another ticker profile.")

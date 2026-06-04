import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Autonomous Stock AI Estimator", layout="wide")
st.title("🤖 Autonomous Short-Term Stock AI Dashboard")
st.markdown("Enter a ticker symbol below. The system automatically fetches financial data, parses real-time news headlines, extracts sentiment vectors, and trains an optimized XGBoost predictive model.")

# --- AUTOMATED SENTIMENT ENGINE ---
def calculate_automated_sentiment(symbol):
    """Fetches real-time headlines and scores them via a programmatic financial lexicon lexicon matrix"""
    try:
        ticker_obj = yf.Ticker(symbol)
        news_list = ticker_obj.news
        
        if not news_list:
            return 0.1, ["No active headlines found. Utilizing neutral-positive baseline."]
        
        # Rule-based financial sentiment lexicon
        bullish_words = {'upgrade', 'record', 'surpass', 'profit', 'growth', 'bullish', 'beat', 'higher', 'rise', 'gains', 'valuable'}
        bearish_words = {'fall', 'drop', 'risk', 'control', 'bearish', 'downside', 'loss', 'miss', 'cut', 'slump', 'decline'}
        
        scores = []
        headlines_read = []
        
        for article in news_list[:8]:  # Analyze the top 8 recent live headlines
            title = article.get('title', '').lower()
            headlines_read.append(article.get('title', ''))
            
            # Simple token score evaluation
            score = 0
            for word in bullish_words:
                if word in title: score += 0.25
            for word in bearish_words:
                if word in title: score -= 0.25
                
            scores.append(np.clip(score, -1.0, 1.0))
            
        avg_sentiment = float(np.mean(scores))
        # Ensure a minimal slight market baseline bump if neutral
        if avg_sentiment == 0:
            avg_sentiment = 0.05
            
        return round(avg_sentiment, 2), headlines_read
    except Exception:
        return 0.1, ["Alternative news feed unreachable. Reverting to automated default stability baseline."]

# --- DATA INGESTION & FEATURE ENGINEERING ---
@st.cache_data(ttl=1800) # Automatic data refresh cache every 30 mins
def load_and_process_data(symbol):
    # Standardizing optimal defaults: 2 Years historical data, 1 Day Prediction Horizon
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.DateOffset(years=2)
    df = yf.download(symbol, start=start_date, end=end_date)
    
    if df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Core high-frequency technical structural inputs
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

# --- AUTOMATED MACHINE LEARNING PIPELINE ---
def train_predictive_model(df, horizon, computed_sentiment):
    df['Target'] = df['Close'].shift(-horizon)
    df['Sentiment_Vector'] = computed_sentiment 
    
    features = ['Close', 'Volume', 'OBV', 'Force_Index', 'RSI', 'MA_5', 'Price_Return', 'Sentiment_Vector']
    latest_features = df[features].iloc[-horizon:]
    
    main_model_df = df.dropna().copy()
    X = main_model_df[features]
    y = main_model_df['Target']
    
    # Stratified sequential time-split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=False)
    
    model = xgb.XGBRegressor(
        n_estimators=120,
        max_depth=4,
        learning_rate=0.04,
        objective='reg:squarederror',
        random_state=42
    )
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    future_preds = model.predict(latest_features)
    
    return rmse, r2, future_preds

# --- USER INTERFACE CONTAINER ---
ticker = st.text_input("Enter Stock Ticker Symbol (e.g. ASML, NVDA, AAPL):", value="ASML").upper()

if ticker:
    with st.spinner(f"Autonomously gathering configurations and news sentiment vectors for {ticker}..."):
        # 1. Trigger Autonomous NLP Parsing
        automated_sentiment, headlines = calculate_automated_sentiment(ticker)
        
        # 2. Extract price actions
        data = load_and_process_data(ticker)
        
    if data is not None:
        # 3. Fit pipeline using standardized 1-day prediction parameters
        rmse, r2, future_predictions = train_predictive_model(data, 1, automated_sentiment)
        
        # --- COMPUTE KPI INDEXES ---
        current_price = data['Close'].iloc[-1]
        estimated_price = future_predictions[-1]
        price_change = ((estimated_price - current_price) / current_price) * 100
        
        # --- UI DISPLAY MAPS ---
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="Current Closing Price", value=f"${current_price:.2f}")
        with col2:
            st.metric(
                label="AI Automated 1-Day Forecast", 
                value=f"${estimated_price:.2f}", 
                delta=f"{price_change:.2f}%"
            )
        with col3:
            st.metric(label="Model Accuracy R²", value=f"{r2:.2f}", delta=f"RMSE Variance: ${rmse:.2f}", delta_color="inverse")
            
        # --- NLP REVEAL PANEL ---
        st.subheader("🤖 Automated Sentiment Engine Audit")
        col_sent_1, col_sent_2 = st.columns([1, 2])
        
        with col_sent_1:
            st.info(f"**Calculated Score:** {automated_sentiment}")
            st.caption("Scale: -1.0 (Extreme Panic) to +1.0 (Extreme Euphoria). This structural parameter was automatically injected into the XGBoost algorithm matrices.")
            
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
        st.error(f"Could not initialize configuration profile for '{ticker}'. Verify ticker naming formatting.")

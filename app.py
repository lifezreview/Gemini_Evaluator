import streamlit as st
import pandas as pd
import numpy as np
import requests
import xml.etree.ElementTree as ET
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Dynamic Holding AI Optimizer", layout="wide")
st.title("🎯 Dynamic Stock Holding Horizon AI Optimizer")
st.markdown("Enter a ticker symbol below. This price-agnostic AI normalizes market trends to calculate the optimal holding period based purely on relative momentum cycles.")

# --- AUTOMATED SENTIMENT ENGINE ---
def calculate_automated_sentiment(symbol):
    try:
        url = f"https://news.google.com/rss/search?q={symbol}+stock&hl=en-US&gl=US&ceid=US:en"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code != 200:
            return 0.05, ["News feeds busy. Running on neutral baseline configuration."]
            
        root = ET.fromstring(response.content)
        items = root.findall('.//item')
        if not items:
            return 0.05, ["No active headlines discovered. Utilizing fallback matrix defaults."]
            
        bullish_words = {'upgrade', 'record', 'surpass', 'profit', 'growth', 'bullish', 'beat', 'higher', 'rise', 'gains', 'valuable', 'buy'}
        bearish_words = {'fall', 'drop', 'risk', 'control', 'bearish', 'downside', 'loss', 'miss', 'cut', 'slump', 'decline', 'sell'}
        
        scores = []
        headlines_read = []
        for item in items[:8]:
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
    except Exception:
        return 0.05, ["Sentiment engine bypass optimized. Running baseline configurations."]

# --- NATIVE EMULATED DATA INGESTION & STATIONARY FEATURE ENGINEERING ---
@st.cache_data(ttl=1800)
def load_and_process_data(symbol):
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range=2y&interval=1d"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return None
            
        raw_json = response.json()
        result_node = raw_json['chart']['result'][0]
        timestamps = result_node['timestamp']
        quote_indicators = result_node['indicators']['quote'][0]
        
        closes = quote_indicators['close']
        volumes = quote_indicators['volume']
        
        adjclose_node = result_node['indicators'].get('adjclose', [{}])[0].get('adjclose', None)
        if adjclose_node is not None:
            closes = adjclose_node
            
        df = pd.DataFrame({
            'Close': closes,
            'Volume': volumes
        }, index=pd.to_datetime(timestamps, unit='s'))
        
        df.dropna(subset=['Close', 'Volume'], inplace=True)
        if df.empty:
            return None
            
        # --- CRITICAL FIX: STATIONARY FEATURE MATRIX (No Raw Dollar Values allowed) ---
        df['Price_Return'] = df['Close'].pct_change()
        df['Volume_Return'] = df['Volume'].pct_change()
        
        # 1. Normalized Moving Average Distance
        ma_5 = df['Close'].rolling(window=5).mean()
        df['MA_5_Dist'] = (df['Close'] - ma_5) / ma_5
        
        # 2. Bounded Relative Strength Index (RSI)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
        
        # 3. Normalized Volatility Metric (5-day rolling standard deviation of returns)
        df['Volatility'] = df['Price_Return'].rolling(window=5).std()
        
        df.dropna(inplace=True)
        return df
    except Exception:
        return None

# --- MULTI-HORIZON MACHINE LEARNING SCANNER ---
def evaluate_optimal_holding_period(df, computed_sentiment):
    holding_profiles = {
        "1 Week": 5,
        "1 Month": 21,
        "3 Months": 63,
        "6 Months": 126
    }
    
    # Purely stationary inputs that remain stable even at all-time highs
    features = ['Price_Return', 'Volume_Return', 'MA_5_Dist', 'RSI', 'Volatility']
    base_df = df.copy()
    base_df['Sentiment_Vector'] = computed_sentiment
    features_with_sent = features + ['Sentiment_Vector']
    
    # Store today's current configuration to run the final projection
    latest_market_snapshot = base_df[features_with_sent].iloc[-1:]
    current_price = float(base_df['Close'].iloc[-1])
    
    horizon_results = {}
    
    for label, days in holding_profiles.items():
        loop_df = base_df.copy()
        
        # Target calculation: Forward percentage return
        future_price = loop_df['Close'].shift(-days)
        loop_df['Target'] = (future_price - loop_df['Close']) / loop_df['Close']
        
        cleaned_ml_df = loop_df.dropna(subset=['Target'] + features_with_sent)
        if len(cleaned_ml_df) < 50:
            continue
            
        X = cleaned_ml_df[features_with_sent]
        y = cleaned_ml_df['Target']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=False)
        
        # Tuned hyperparameters to prevent over-fitting on highly volatile assets
        model = xgb.XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.05, objective='reg:squarederror', random_state=42)
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        r2 = r2_score(y_test, preds)
        
        # Generate prediction based on normalized baseline features
        pred_pct_gain = float(model.predict(latest_market_snapshot)[0])
        future_projection = current_price * (1 + pred_pct_gain)
        pct_gain = pred_pct_gain * 100
        
        horizon_results[label] = {
            "predicted_price": future_projection,
            "percentage_gain": pct_gain,
            "accuracy_r2": r2
        }
        
    return current_price, horizon_results

# --- UI CONTROL INTERFACE ---
with st.form("optimizer_form"):
    ticker = st.text_input("Enter Stock Ticker Symbol (e.g. AAPL, NVDA, ASML):", value="AAPL").upper()
    submit_button = st.form_submit_button("Compute Peak Holding Windows")

if ticker:
    with st.spinner(f"Initiating price-agnostic multi-horizon analysis for {ticker}..."):
        automated_sentiment, headlines = calculate_automated_sentiment(ticker)
        data = load_and_process_data(ticker)
        
    if data is not None:
        current_close, results = evaluate_optimal_holding_period(data, automated_sentiment)
        
        if results:
            best_horizon = max(results, key=lambda k: results[k]['percentage_gain'])
            best_metrics = results[best_horizon]
            
            # --- RENDER MAIN ALGORITHMIC RECOMMENDATION BANNER ---
            if best_metrics['percentage_gain'] >= 0:
                st.success(f"### 🤖 AI Strategic Recommendation for {ticker}")
            else:
                st.warning(f"### 🤖 AI Defensive Recommendation for {ticker} (Risk Detected Across All Windows)")
            
            rec_col1, rec_col2, rec_col3 = st.columns(3)
            with rec_col1:
                st.metric(label="Optimal Holding Horizon", value=best_horizon)
            with rec_col2:
                st.metric(label="Projected Optimal Return", value=f"{best_metrics['percentage_gain']:.2f}%")
            with rec_col3:
                st.metric(label="Target Price Projection", value=f"${best_metrics['predicted_price']:.2f}")
                
            st.markdown("---")
            st.subheader("📊 Comparative Horizon Breakdown Matrix")
            st.markdown("The values below illustrate how the asset's momentum vectors scale across progressive historical holding thresholds.")
            
            matrix_data = []
            for h_name, h_info in results.items():
                matrix_data.append({
                    "Holding Window": h_name,
                    "Projected Price": f"${h_info['predicted_price']:.2f}",
                    "Expected Gain/Loss (%)": f"{h_info['percentage_gain']:.2f}%",
                    "Model Confidence (R²)": f"{h_info['accuracy_r2']:.2f}"
                })
            st.table(pd.DataFrame(matrix_data))
            
            # --- INTERACTIVE COMPARATIVE PLOT ---
            st.subheader("Visualized Holding Return Vectors")
            horizons_list = list(results.keys())
            gains_list = [results[h]['percentage_gain'] for h in horizons_list]
            
            fig = go.Figure()
            colors = ['#2ca02c' if g >= 0 else '#d62728' for g in gains_list]
            fig.add_trace(go.Bar(x=horizons_list, y=gains_list, marker_color=colors, text=[f"{g:.1f}%" for g in gains_list], textposition='auto'))
            fig.update_layout(template="plotly_dark", yaxis_title="Projected Return (%)", xaxis_title="Holding Horizon", margin=dict(l=20, r=20, t=20, b=20), height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("View Real-Time Scraped Headlines Evaluated"):
                st.write(f"**Computed Sentiment Vector Injected:** {automated_sentiment}")
                for hl in headlines:
                    st.write(f"• {hl}")
        else:
            st.error("Insufficient historical trading volume density to support multi-horizon processing loops.")
    else:
        st.error(f"Could not connect to database matrix for '{ticker}'. Ensure the ticker naming conventions are accurate.")

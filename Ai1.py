import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import feedparser
import urllib.parse
import requests
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from xgboost import XGBClassifier
import nltk

# Download free NLTK sentiment lexicon
nltk.download('vader_lexicon', quiet=True)

# Streamlit Page Config
st.set_page_config(
    page_title="AI Market Intelligence Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM MODERN DARK UI STYLING ---
st.markdown("""
<style>
    .main {
        background-color: #0E1117;
    }
    .metric-card {
        background: linear-gradient(135deg, #1E2640 0%, #111827 100%);
        border: 1px solid #374151;
        border-radius: 12px;
        padding: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.5);
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        margin-top: 5px;
    }
    .metric-label {
        font-size: 14px;
        color: #9CA3AF;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .signal-bullish {
        background-color: rgba(16, 185, 129, 0.1);
        border: 1px solid #10B981;
        color: #34D399;
        padding: 15px;
        border-radius: 10px;
        font-weight: bold;
        font-size: 20px;
        text-align: center;
    }
    .signal-bearish {
        background-color: rgba(239, 68, 68, 0.1);
        border: 1px solid #EF4444;
        color: #F87171;
        padding: 15px;
        border-radius: 10px;
        font-weight: bold;
        font-size: 20px;
        text-align: center;
    }
    .signal-neutral {
        background-color: rgba(245, 158, 11, 0.1);
        border: 1px solid #F59E0B;
        color: #FBBF24;
        padding: 15px;
        border-radius: 10px;
        font-weight: bold;
        font-size: 20px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.title("⚡ Pro AI Market Intelligence & Predictive Signal Engine")
st.caption("Powered by XGBoost Machine Learning, Global NLP Sentiment Analysis & Quantitative Analytics")

# --- COMPANY NAME TO TICKER SEARCH FUNCTION ---
@st.cache_data(ttl=86400)
def search_company_ticker(query):
    """Searches Yahoo Finance API for tickers based on company name."""
    if not query or len(query) < 2:
        return []
    try:
        url = "https://query2.finance.yahoo.com/v1/finance/search"
        headers = {'User-Agent': 'Mozilla/5.0'}
        params = {'q': query, 'quotesCount': 8, 'newsCount': 0}
        response = requests.get(url, headers=headers, params=params, timeout=5)
        data = response.json()
        
        results = []
        if 'quotes' in data:
            for quote in data['quotes']:
                if 'symbol' in quote and 'shortname' in quote:
                    exch = quote.get('exchDisp', 'Global')
                    results.append(f"{quote['shortname']} ({quote['symbol']}) — {exch}")
                elif 'symbol' in quote:
                    results.append(f"{quote['symbol']} ({quote['symbol']})")
        return results
    except Exception:
        return []

# --- CACHED MARKET & NEWS DATA FETCHING ---
@st.cache_data(ttl=3600)
def fetch_stock_data(symbol):
    try:
        df = yf.download(symbol, period="365d", interval="1d", progress=False)
        if df.empty:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Quantitative Indicators
        df['Returns'] = df['Close'].pct_change()
        
        # 14-Day RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Moving Averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        # Volatility & Volume change
        df['Volatility'] = df['Returns'].rolling(window=20).std()
        df['Volume_Change'] = df['Volume'].pct_change()

        return df
    except Exception:
        return None

@st.cache_data(ttl=1800)
def fetch_rss_news(query_term):
    try:
        encoded = urllib.parse.quote(f"{query_term} stock market news")
        feed_url = f"https://news.google.com/rss/search?q={encoded}&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(feed_url)
        
        headlines = []
        for entry in feed.entries[:6]:
            headlines.append({
                "title": entry.title,
                "link": entry.link,
                "published": entry.published if hasattr(entry, 'published') else "Recent"
            })
        return headlines
    except Exception:
        return []

# --- SIDEBAR & SEARCH INTERFACE ---
with st.sidebar:
    st.header("🔍 Stock Lookup")
    search_input = st.text_input("Type Company Name:", "Nvidia")
    
    ticker_options = search_company_ticker(search_input)
    
    if ticker_options:
        selected_option = st.selectbox("Select Matching Company:", ticker_options)
        # Extract ticker symbol from parentheses
        selected_ticker = selected_option.split("(")[-1].split(")")[0]
    else:
        st.info("Searching for ticker or type symbol manually below...")
        selected_ticker = st.text_input("Manual Ticker Code:", "NVDA").upper()

    st.markdown("---")
    st.markdown("### ⚙️ Engine Settings")
    prediction_horizon = st.slider("Prediction Target Window:", 1, 10, 3, help="Days ahead for model forecast")

# --- MAIN ENGINE RUN ---
if st.button("🚀 Execute AI Market Analysis", type="primary", use_container_width=True):
    with st.spinner(f"Training XGBoost Engine & Scraping Global News Flow for {selected_ticker}..."):
        df = fetch_stock_data(selected_ticker)
        
        if df is None or len(df) < 60:
            st.error("Could not fetch sufficient market data for this stock symbol.")
        else:
            news_items = fetch_rss_news(selected_ticker)
            
            # --- NLP NEWS SENTIMENT ENGINE ---
            sia = SentimentIntensityAnalyzer()
            sentiment_scores = []
            for item in news_items:
                s = sia.polarity_scores(item['title'])['compound']
                sentiment_scores.append(s)
            
            avg_sentiment = float(np.mean(sentiment_scores)) if sentiment_scores else 0.0
            
            # --- ADVANCED XGBOOST MODEL TRAINING ---
            df['News_Sentiment'] = avg_sentiment
            df['Target'] = (df['Close'].shift(-prediction_horizon) > df['Close']).astype(int)
            
            features = ['Returns', 'RSI', 'Volatility', 'Volume_Change', 'News_Sentiment']
            clean_df = df.dropna(subset=features + ['Target'])
            
            X = clean_df[features]
            y = clean_df['Target']
            
            # Train model on historical sequences
            model = XGBClassifier(n_estimators=40, max_depth=3, learning_rate=0.05, eval_metric='logloss')
            model.fit(X[:-1], y[:-1])
            
            # Predict on latest live state
            latest_features = X.iloc[[-1]]
            boost_prob = float(model.predict_proba(latest_features)[0][1]) * 100
            
            # Metric Extract
            latest_price = float(df['Close'].iloc[-1])
            latest_rsi = float(df['RSI'].iloc[-1]) if not pd.isna(df['RSI'].iloc[-1]) else 50.0
            price_change = float(df['Returns'].iloc[-1]) * 100
            
            # --- DASHBOARD LAYOUT ---
            st.markdown("### 📊 Market Flow Indicators")
            c1, c2, c3, c4 = st.columns(4)
            
            with c1:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Current Price</div><div class="metric-value">${latest_price:.2f} ({price_change:+.2f}%)</div></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="metric-card"><div class="metric-label">RSI (14-Day)</div><div class="metric-value">{latest_rsi:.1f}</div></div>', unsafe_allow_html=True)
            with c3:
                st.markdown(f'<div class="metric-card"><div class="metric-label">News Sentiment</div><div class="metric-value">{avg_sentiment:+.2f}</div></div>', unsafe_allow_html=True)
            with c4:
                st.markdown(f'<div class="metric-card"><div class="metric-label">AI Boost Forecast</div><div class="metric-value">{boost_prob:.1f}%</div></div>', unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # --- PREDICTION BANNER ---
            if boost_prob >= 60.0:
                st.markdown(f'<div class="signal-bullish">🚀 STRONG BUY / BOOST SIGNAL — {boost_prob:.1f}% AI Probability of Price Increase in {prediction_horizon} Days</div>', unsafe_allow_html=True)
            elif boost_prob <= 40.0:
                st.markdown(f'<div class="signal-bearish">⚠️ BEARISH / RISK SIGNAL — High Risk of Pullback ({boost_prob:.1f}% Growth Probability)</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="signal-neutral">⚖️ NEUTRAL / CONSOLIDATION — Market Flow is Range-Bound ({boost_prob:.1f}% Probability)</div>', unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # --- CHARTS ---
            col_chart, col_news = st.columns([2, 1])
            
            with col_chart:
                st.subheader("📈 Price Action & Trend Analysis")
                st.line_chart(df[['Close', 'SMA_20', 'SMA_50']])
                
            with col_news:
                st.subheader("🌍 Live Global News Feed")
                if news_items:
                    for item, score in zip(news_items, sentiment_scores):
                        if score > 0.05:
                            badge = "🟢 POSITIVE"
                        elif score < -0.05:
                            badge = "🔴 NEGATIVE"
                        else:
                            badge = "⚪ NEUTRAL"
                        st.markdown(f"**[{badge}]** [{item['title']}]({item['link']})")
                        st.caption(f"Published: {item['published']}")
                        st.markdown("---")
                else:
                    st.info("No recent news feed available.")

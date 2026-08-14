import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import feedparser
import urllib.parse
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download free sentiment lexicon
nltk.download('vader_lexicon', quiet=True)

st.set_page_config(page_title="Free AI Stock & News Predictor", layout="wide")
st.title("📈 AI Stock & Global News Analyzer")
st.caption("Powered by Open-Source Sentiment AI & Quantitative Analysis ($0 Cost Stack)")

# --- CACHED DATA FETCHING (Prevents Rate-Limit Errors) ---
@st.cache_data(ttl=3600)  # Cache results for 1 hour (3600 seconds)
def fetch_stock_data(symbol):
    try:
        # Download history directly without touching stock.info (prevents YFRateLimitError)
        df = yf.download(symbol, period="180d", interval="1d", progress=False)
        
        if df.empty:
            return None

        # Fix multi-index columns if yfinance returns them
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Calculate Technical Indicators
        df['Returns'] = df['Close'].pct_change()
        
        # 14-Day Relative Strength Index (RSI)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 20-Day Simple Moving Average
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        
        return df
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return None

@st.cache_data(ttl=1800)  # Cache news for 30 minutes
def fetch_rss_news(symbol):
    try:
        # Free Google News RSS Feed
        query = f"{symbol} stock market news"
        encoded_query = urllib.parse.quote(query)
        feed_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(feed_url)
        
        headlines = []
        for entry in feed.entries[:5]:  # Top 5 headlines
            headlines.append({
                "title": entry.title,
                "link": entry.link,
                "published": entry.published if hasattr(entry, 'published') else "Recent"
            })
        return headlines
    except Exception as e:
        return []

# --- USER INPUT SECTION ---
ticker = st.text_input("Enter Stock Ticker (e.g., NVDA, AAPL, TSLA, MSFT):", "NVDA").upper().strip()

if st.button("Run AI Market Flow Analysis"):
    if not ticker:
        st.warning("Please enter a valid stock ticker.")
    else:
        with st.spinner(f"Analyzing price flow & global news sentiment for {ticker}..."):
            df = fetch_stock_data(ticker)
            
            if df is None or len(df) < 20:
                st.error("Unable to retrieve stock data or ticker symbol is invalid. Please double-check the symbol.")
            else:
                news_items = fetch_rss_news(ticker)
                
                # --- SENTIMENT ANALYSIS ---
                sia = SentimentIntensityAnalyzer()
                sentiment_scores = []
                for item in news_items:
                    score = sia.polarity_scores(item['title'])['compound']
                    sentiment_scores.append(score)
                
                avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0
                
                # --- CALCULATE PREDICTIVE BOOST SIGNAL ---
                latest_price = float(df['Close'].iloc[-1])
                latest_rsi = float(df['RSI'].iloc[-1]) if not pd.isna(df['RSI'].iloc[-1]) else 50.0
                monthly_return = float((df['Close'].iloc[-1] - df['Close'].iloc[-20]) / df['Close'].iloc[-20])
                
                # RSI Score Normalization
                if 40 <= latest_rsi <= 65:
                    rsi_signal = 0.5   # Healthy momentum
                elif latest_rsi > 70:
                    rsi_signal = -0.5  # Overbought warning
                else:
                    rsi_signal = 0.2   # Oversold bounce potential
                
                # Composite Boost Formula (40% News Sentiment + 40% RSI Momentum + 20% Monthly Return)
                boost_score = (avg_sentiment * 0.4) + (rsi_signal * 0.4) + (monthly_return * 0.2)
                
                # --- DISPLAY METRICS ---
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Current Price", f"${latest_price:.2f}")
                col2.metric("RSI (14-Day)", f"{latest_rsi:.1f}")
                col3.metric("News Sentiment", f"{avg_sentiment:+.2f}")
                col4.metric("AI Boost Score", f"{boost_score:+.2f}")
                
                st.markdown("---")
                
                # Signal Output Banner
                if boost_score > 0.15:
                    st.success("🚀 **Signal: High Probability of Upward Boost**")
                elif boost_score < -0.15:
                    st.error("⚠️ **Signal: Bearish / Risk of Downward Pullback**")
                else:
                    st.warning("⚖️ **Signal: Neutral / Consolidation Flow**")
                
                # --- CHART DISPLAY ---
                st.subheader("Market Flow: Closing Price vs 20-Day Moving Average")
                st.line_chart(df[['Close', 'SMA_20']])
                
                # --- NEWS HEADLINES DISPLAY ---
                st.subheader("Global News Sentiment Feed")
                if news_items:
                    for item, s_score in zip(news_items, sentiment_scores):
                        if s_score > 0.05:
                            tag = "🟢 POSITIVE"
                        elif s_score < -0.05:
                            tag = "🔴 NEGATIVE"
                        else:
                            tag = "⚪ NEUTRAL"
                        
                        st.markdown(f"- **[{tag}]** [{item['title']}]({item['link']})")
                else:
                    st.info("No recent news headlines available for this ticker.")

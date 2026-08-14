import streamlit as st
import yfinance as yf
import pandas as pd
import feedparser
import urllib.parse
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download free sentiment lexicon
nltk.download('vader_lexicon', quiet=True)

st.set_page_config(page_title="Free AI Stock Screener", layout="wide")
st.title("📈 AI Stock & News Analyzer ($0 Cost Stack)")

# User Controls
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, NVDA, TSLA):", "AAPL").upper()

def get_technical_data(symbol):
    stock = yf.Ticker(symbol)
    df = stock.history(period="60d")
    if df.empty:
        return None, None
    
    # Calculate 14-day RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Simple Moving Average
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    return stock.info, df

def fetch_rss_news(query):
    # Free Google News RSS Feed
    encoded_query = urllib.parse.quote(f"{query} stock market")
    feed_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(feed_url)
    
    headlines = []
    for entry in feed.entries[:5]:  # Top 5 headlines
        headlines.append({"title": entry.title, "link": entry.link, "published": entry.published})
    return headlines

if st.button("Run AI Analysis"):
    with st.spinner("Fetching data and running algorithms..."):
        info, df = get_technical_data(ticker)
        
        if df is None:
            st.error("Invalid ticker or data unavailable.")
        else:
            latest_price = df['Close'].iloc[-1]
            rsi = df['RSI'].iloc[-1]
            
            # Fetch & Analyze News
            company_name = info.get('shortName', ticker)
            news_items = fetch_rss_news(company_name)
            
            sia = SentimentIntensityAnalyzer()
            sentiment_scores = []
            
            for item in news_items:
                score = sia.polarity_scores(item['title'])['compound']
                sentiment_scores.append(score)
            
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
            
            # Simple Prediction Algorithm
            # Weights: 40% Sentiment, 40% Technical RSI Momentum, 20% Price Trend
            monthly_return = (df['Close'].iloc[-1] - df['Close'].iloc[-20]) / df['Close'].iloc[-20]
            rsi_score = 1.0 if 40 <= rsi <= 65 else (-0.5 if rsi > 70 else 0.5)
            
            composite_signal = (avg_sentiment * 0.4) + (rsi_score * 0.4) + (monthly_return * 0.2)
            
            # UI Metrics Display
            col1, col2, col3 = st.columns(3)
            col1.metric("Current Price", f"${latest_price:.2f}")
            col2.metric("Relative Strength (RSI)", f"{rsi:.1f}")
            col3.metric("AI Boost Score", f"{composite_signal:.2f}")
            
            # Recommendation Banner
            if composite_signal > 0.2:
                st.success("🚀 **Signal: Bullish / Growth Potential**")
            elif composite_signal < -0.1:
                st.error("⚠️ **Signal: Bearish / Downtrend Risk**")
            else:
                st.warning("⚖️ **Signal: Neutral / Hold**")
                
            # Charting
            st.subheader("Price History & Moving Averages")
            st.line_chart(df[['Close', 'SMA_20']])
            
            # Recent News Headlines
            st.subheader("Latest Global News Headlines")
            for item, s_score in zip(news_items, sentiment_scores):
                sentiment_label = "🟢 Positive" if s_score > 0.05 else ("🔴 Negative" if s_score < -0.05 else "⚪ Neutral")
                st.write(f"- [{item['title']}]({item['link']}) — **{sentiment_label}** ({s_score})")

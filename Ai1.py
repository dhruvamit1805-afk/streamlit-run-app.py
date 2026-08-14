import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import feedparser
import urllib.parse
import requests
import re
import json
from datetime import datetime, timedelta
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from xgboost import XGBClassifier
from groq import Groq
import shap
import altair as alt
import nltk

# NLTK Setup
nltk.download('vader_lexicon', quiet=True)

# Streamlit Page Config for Ultra-Sleek UI
st.set_page_config(
    page_title="AI Market Intelligence Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- MAGICAL ANIMATIONS & CUSTOM UI STYLING ---
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    body {{
        font-family: 'Inter', sans-serif;
        background-color: #0E1117;
    }}
    
    .stApp {{
        background: radial-gradient(circle at 10% 20%, rgba(17, 24, 39, 1) 0%, rgba(14, 17, 23, 1) 90%);
    }}
    
    /* Magically Animated Metric Cards */
    .metric-card {{
        background: rgba(31, 41, 55, 0.6);
        border: 1px solid rgba(75, 85, 99, 0.4);
        border-radius: 16px;
        padding: 24px;
        transition: all 0.4s ease-in-out;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(5px);
        margin-bottom: 15px;
    }}
    .metric-card:hover {{
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.5), 0 0 15px rgba(59, 130, 246, 0.3);
        border: 1px solid rgba(59, 130, 246, 0.6);
    }}
    
    .metric-value {{
        font-size: 34px;
        font-weight: 700;
        background: linear-gradient(90deg, #FFFFFF, #9CA3AF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    .metric-label {{
        font-size: 14px;
        color: #9CA3AF;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 600;
    }}
    
    /* Advanced Decision Gauge */
    .gauge-container {{
        width: 100%;
        background-color: #1F2937;
        border-radius: 20px;
        position: relative;
        height: 40px;
        border: 2px solid #374151;
        overflow: hidden;
    }}
    
    .gauge-bar {{
        height: 100%;
        border-radius: 18px;
        transition: width 1.5s ease-out;
        background: linear-gradient(90deg, #10B981, #FBBF24, #EF4444);
        background-size: 300% 100%;
    }}
    
    .gauge-marker {{
        position: absolute;
        width: 4px;
        height: 100%;
        background-color: white;
        top: 0;
        transition: left 1.5s ease-out;
    }}

</style>
""", unsafe_allow_html=True)

st.title("⚡ Pro AI Quantitative Advisor & Momentum Engine")
st.caption("Fusing Machine Learning, LLM Logic, and Stochastic Simulations ($0 Cost)")

# --- UTILITIES ---
def search_company_ticker(query):
    if not query or len(query) < 2: return []
    try:
        url = "https://query2.finance.yahoo.com/v1/finance/search"
        data = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, params={'q': query, 'quotesCount': 8}).json()
        return [f"{q['shortname']} ({q['symbol']})" for q in data.get('quotes', []) if 'symbol' in q and 'shortname' in q]
    except Exception: return []

@st.cache_data(ttl=3600)
def fetch_data(symbol):
    try:
        df = yf.download(symbol, period="2y", interval="1d", progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        
        # Basic indicators
        df['Returns'] = df['Close'].pct_change()
        df['RSI'] = 100 - (100 / (1 + (df['Close'].diff().where(df['Close'].diff() > 0, 0).rolling(14).mean() / (-df['Close'].diff().where(df['Close'].diff() < 0, 0).rolling(14).mean()))))
        df['SMA_20'], df['SMA_50'] = df['Close'].rolling(20).mean(), df['Close'].rolling(50).mean()
        df['Volatility'], df['Volume_Change'] = df['Returns'].rolling(20).std(), df['Volume'].pct_change()
        return df.dropna()
    except Exception: return None

@st.cache_data(ttl=1800)
def fetch_news_sentiment(ticker):
    try:
        feed = feedparser.parse(f"https://news.google.com/rss/search?q={urllib.parse.quote(f'{ticker} stock')}&hl=en-US")
        titles = [e.title for e in feed.entries[:6]]
        sia = SentimentIntensityAnalyzer()
        scores = [sia.polarity_scores(t)['compound'] for t in titles]
        return float(np.mean(scores)) if scores else 0.0, titles, scores
    except Exception: return 0.0, [], []

# --- ADVANCED AI ROUTINES ---
def run_xgb_model(df, sentiment_score, prediction_horizon):
    df['News_Sentiment'] = sentiment_score
    df['Target'] = (df['Close'].shift(-prediction_horizon) > df['Close']).astype(int)
    
    features = ['Returns', 'RSI', 'Volatility', 'Volume_Change', 'News_Sentiment']
    clean_df = df.dropna(subset=features + ['Target'])
    X, y = clean_df[features], clean_df['Target']
    
    model = XGBClassifier(n_estimators=40, max_depth=3, learning_rate=0.05, eval_metric='logloss')
    model.fit(X[:-1], y[:-1])
    
    latest_feat = X.iloc[[-1]]
    boost_prob = float(model.predict_proba(latest_feat)[0][1]) * 100
    
    # SHAP Explainability
    explainer = shap.Explainer(model, X[:-1])
    shap_values = explainer(latest_feat)
    
    return boost_prob, shap_values, features

@st.cache_data(ttl=1800)
def get_llm_summary(ticker, price, rsi, sentiment, xgb_prob, shap_contrib, headlines):
    try:
        if not st.secrets.get("GROQ_API_KEY"): return "Error: Groq API Key missing in secrets."
        
        client = Groq(api_key=st.secrets["GROQ_API_KEY"])
        prompt = f"""
        Act as a senior Quantitative Stock Analyst.
        Stock: {ticker}, Current Price: ${price:.2f}.
        Technical RSI (14-day): {rsi:.1f}.
        NLP News Sentiment: {sentiment:.2f}.
        Advanced XGBoost Prediction Probability (Boost in 3-5 days): {xgb_prob:.1f}%.
        AI Decision Drivers (SHAP contributions): {json.dumps(shap_contrib)}.
        Top News Headlines: {json.dumps(headlines[:3])}.

        Provide an urgent, sleek executive advice summary in exactly 3 bullet points:
        1. Current Momentum Driver (Technicals vs Sentiment)
        2. Key Risk Factor
        3. The Final 'Go/No-Go' Verdict based on the AI probability.
        Keep it aggressive, intelligent, and professional. Use markdown bolding.
        """
        response = client.chat.completions.create(model="llama3-70b-8192", messages=[{"role": "user", "content": prompt}], temperature=0.2)
        return response.choices[0].message.content
    except Exception: return "Advisor service temporarily unavailable."

# --- SIMULATION ROUTINES ---
def run_monte_carlo(df, days=30, sims=1000):
    latest_price = df['Close'].iloc[-1]
    returns = df['Returns'].dropna()
    mu = returns.mean()
    sigma = returns.std()
    
    # Simulate future paths using Geometric Brownian Motion
    sim_results = np.zeros((days, sims))
    for s in range(sims):
        sim_prices = [latest_price]
        for d in range(1, days):
            sim_prices.append(sim_prices[-1] * np.exp(mu + sigma * np.random.normal()))
        sim_results[:, s] = sim_prices
    
    return sim_results

# --- UI INTERFACE ---
with st.sidebar:
    st.header("🔍 Intelligent Search")
    search_input = st.text_input("Find Company:", "Microsoft")
    ticker_options = search_company_ticker(search_input)
    
    if ticker_options: selected_option = st.selectbox("Select Target:", ticker_options)
    else: selected_option = st.selectbox("Defaults:", ["Microsoft (MSFT)", "Apple (AAPL)", "Nvidia (NVDA)"])
    selected_ticker = selected_option.split("(")[-1].split(")")[0]
    
    st.markdown("---")
    days_to_sim = st.slider("Future Sim Window (Days)", 10, 90, 30)

# --- EXECUTION ---
if st.button("🚀 EXECUTE QUANTITATIVE COMMAND", type="primary", use_container_width=True):
    with st.spinner(f"Initiating Neural Models and Simulations for {selected_ticker}..."):
        df = fetch_data(selected_ticker)
        if df is None: st.error("Execution failure: Insufficient market flow data."); st.stop()
        
        # Calculate AI Components
        avg_sentiment, headlines, sentiment_scores = fetch_news_sentiment(selected_ticker)
        xgb_prob, shap_values, features = run_xgb_model(df, avg_sentiment, 3)
        sim_paths = run_monte_carlo(df, days=days_to_sim)
        
        # Generate SHAP dictionary
        shap_contrib = {feat: float(val) for feat, val in zip(features, shap_values.values[0])}
        
        # Fetch LLM Summary
        llm_advice = get_llm_summary(selected_ticker, df['Close'].iloc[-1], df['RSI'].iloc[-1], avg_sentiment, xgb_prob, shap_contrib, headlines)

        # Dashboard Metrics
        latest = df.iloc[-1]
        change = float(df['Returns'].iloc[-1]) * 100
        
        # 1. SLEEK METRIC CARDS
        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(f'<div class="metric-card"><div class="metric-label">Live Price</div><div class="metric-value">${latest["Close"]:.2f} ({change:+.2f}%)</div></div>', unsafe_allow_html=True)
        m2.markdown(f'<div class="metric-card"><div class="metric-label">RSI (14-Day)This is a massive upgrade that transforms your app into a **professional-grade AI quantitative dashboard** with **magical animations, explainable AI, future simulations, and an LLM advisor**.

### Crucial: Pre-Setup ($0 Cost)

To make this app work, you **must** get a free API key from **Groq** to power the LLaMA-3 executive advisor.

1.  Go to **[console.groq.com](https://console.groq.com/)** and sign up for a free account.
2.  Create an API Key and copy it.
3.  Add it to your Streamlit secrets on Streamlit Cloud (Settings $\rightarrow$ Secrets):
    ```toml
    GROQ_API_KEY = "your_key_here_xxxx"
    ```

---

### Replace EVERYTHING in `Ai1.py` with this master code:

```python
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import feedparser
import urllib.parse
import requests
import re
import json
from datetime import datetime, timedelta
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from xgboost import XGBClassifier
from groq import Groq
import shap
import altair as alt
import nltk

# NLTK Setup
nltk.download('vader_lexicon', quiet=True)

# Streamlit Page Config for Ultra-Sleek UI
st.set_page_config(
    page_title="AI Market Intelligence Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- MAGICAL ANIMATIONS & CUSTOM UI STYLING ---
st.markdown(f"""
<style>
    @import url('[https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap](https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap)');
    
    body {{
        font-family: 'Inter', sans-serif;
        background-color: #0E1117;
    }}
    
    .stApp {{
        background: radial-gradient(circle at 10% 20%, rgba(17, 24, 39, 1) 0%, rgba(14, 17, 23, 1) 90%);
    }}
    
    /* Magically Animated Metric Cards */
    .metric-card {{
        background: rgba(31, 41, 55, 0.6);
        border: 1px solid rgba(75, 85, 99, 0.4);
        border-radius: 16px;
        padding: 24px;
        transition: all 0.4s ease-in-out;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(5px);
        margin-bottom: 15px;
    }}
    .metric-card:hover {{
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.5), 0 0 15px rgba(59, 130, 246, 0.3);
        border: 1px solid rgba(59, 130, 246, 0.6);
    }}
    
    .metric-value {{
        font-size: 34px;
        font-weight: 700;
        background: linear-gradient(90deg, #FFFFFF, #9CA3AF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    .metric-label {{
        font-size: 14px;
        color: #9CA3AF;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 600;
    }}
    
    /* Advanced Decision Gauge */
    .gauge-container {{
        width: 100%;
        background-color: #1F2937;
        border-radius: 20px;
        position: relative;
        height: 40px;
        border: 2px solid #374151;
        overflow: hidden;
    }}
    
    .gauge-bar {{
        height: 100%;
        border-radius: 18px;
        transition: width 1.5s ease-out;
        background: linear-gradient(90deg, #10B981, #FBBF24, #EF4444);
        background-size: 300% 100%;
    }}
    
    .gauge-marker {{
        position: absolute;
        width: 4px;
        height: 100%;
        background-color: white;
        top: 0;
        transition: left 1.5s ease-out;
    }}

</style>
""", unsafe_allow_html=True)

st.title("⚡ Pro AI Quantitative Advisor & Momentum Engine")
st.caption("Fusing Machine Learning, LLM Logic, and Stochastic Simulations ($0 Cost)")

# --- UTILITIES ---
def search_company_ticker(query):
    if not query or len(query) < 2: return []
    try:
        url = "[https://query2.finance.yahoo.com/v1/finance/search](https://query2.finance.yahoo.com/v1/finance/search)"
        data = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, params={'q': query, 'quotesCount': 8}).json()
        return [f"{q['shortname']} ({q['symbol']})" for q in data.get('quotes', []) if 'symbol' in q and 'shortname' in q]
    except Exception: return []

@st.cache_data(ttl=3600)
def fetch_data(symbol):
    try:
        df = yf.download(symbol, period="2y", interval="1d", progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        
        # Basic indicators
        df['Returns'] = df['Close'].pct_change()
        df['RSI'] = 100 - (100 / (1 + (df['Close'].diff().where(df['Close'].diff() > 0, 0).rolling(14).mean() / (-df['Close'].diff().where(df['Close'].diff() < 0, 0).rolling(14).mean()))))
        df['SMA_20'], df['SMA_50'] = df['Close'].rolling(20).mean(), df['Close'].rolling(50).mean()
        df['Volatility'], df['Volume_Change'] = df['Returns'].rolling(20).std(), df['Volume'].pct_change()
        return df.dropna()
    except Exception: return None

@st.cache_data(ttl=1800)
def fetch_news_sentiment(ticker):
    try:
        feed = feedparser.parse(f"[https://news.google.com/rss/search?q=](https://news.google.com/rss/search?q=){urllib.parse.quote(f'{ticker} stock')}&hl=en-US")
        titles = [e.title for e in feed.entries[:6]]
        sia = SentimentIntensityAnalyzer()
        scores = [sia.polarity_scores(t)['compound'] for t in titles]
        return float(np.mean(scores)) if scores else 0.0, titles, scores
    except Exception: return 0.0, [], []

# --- ADVANCED AI ROUTINES ---
def run_xgb_model(df, sentiment_score, prediction_horizon):
    df['News_Sentiment'] = sentiment_score
    df['Target'] = (df['Close'].shift(-prediction_horizon) > df['Close']).astype(int)
    
    features = ['Returns', 'RSI', 'Volatility', 'Volume_Change', 'News_Sentiment']
    clean_df = df.dropna(subset=features + ['Target'])
    X, y = clean_df[features], clean_df['Target']
    
    model = XGBClassifier(n_estimators=40, max_depth=3, learning_rate=0.05, eval_metric='logloss')
    model.fit(X[:-1], y[:-1])
    
    latest_feat = X.iloc[[-1]]
    boost_prob = float(model.predict_proba(latest_feat)[0][1]) * 100
    
    # SHAP Explainability
    explainer = shap.Explainer(model, X[:-1])
    shap_values = explainer(latest_feat)
    
    return boost_prob, shap_values, features

@st.cache_data(ttl=1800)
def get_llm_summary(ticker, price, rsi, sentiment, xgb_prob, shap_contrib, headlines):
    try:
        if not st.secrets.get("GROQ_API_KEY"): return "Error: Groq API Key missing in secrets."
        
        client = Groq(api_key=st.secrets["GROQ_API_KEY"])
        prompt = f"""
        Act as a senior Quantitative Stock Analyst.
        Stock: {ticker}, Current Price: ${price:.2f}.
        Technical RSI (14-day): {rsi:.1f}.
        NLP News Sentiment: {sentiment:.2f}.
        Advanced XGBoost Prediction Probability (Boost in 3-5 days): {xgb_prob:.1f}%.
        AI Decision Drivers (SHAP contributions): {json.dumps(shap_contrib)}.
        Top News Headlines: {json.dumps(headlines[:3])}.

        Provide an urgent, sleek executive advice summary in exactly 3 bullet points:
        1. Current Momentum Driver (Technicals vs Sentiment)
        2. Key Risk Factor
        3. The Final 'Go/No-Go' Verdict based on the AI probability.
        Keep it aggressive, intelligent, and professional. Use markdown bolding.
        """
        response = client.chat.completions.create(model="llama3-70b-8192", messages=[{"role": "user", "content": prompt}], temperature=0.2)
        return response.choices[0].message.content
    except Exception: return "Advisor service temporarily unavailable."

# --- SIMULATION ROUTINES ---
def run_monte_carlo(df, days=30, sims=1000):
    latest_price = df['Close'].iloc[-1]
    returns = df['Returns'].dropna()
    mu = returns.mean()
    sigma = returns.std()
    
    # Simulate future paths using Geometric Brownian Motion
    sim_results = np.zeros((days, sims))
    for s in range(sims):
        sim_prices = [latest_price]
        for d in range(1, days):
            sim_prices.append(sim_prices[-1] * np.exp(mu + sigma * np.random.normal()))
        sim_results[:, s] = sim_prices
    
    return sim_results

# --- UI INTERFACE ---
with st.sidebar:
    st.header("🔍 Intelligent Search")
    search_input = st.text_input("Find Company:", "Microsoft")
    ticker_options = search_company_ticker(search_input)
    
    if ticker_options: selected_option = st.selectbox("Select Target:", ticker_options)
    else: selected_option = st.selectbox("Defaults:", ["Microsoft (MSFT)", "Apple (AAPL)", "Nvidia (NVDA)"])
    selected_ticker = selected_option.split("(")[-1].split(")")[0]
    
    st.markdown("---")
    days_to_sim = st.slider("Future Sim Window (Days)", 10, 90, 30)

# --- EXECUTION ---
if st.button("🚀 EXECUTE QUANTITATIVE COMMAND", type="primary", use_container_width=True):
    with st.spinner(f"Initiating Neural Models and Simulations for {selected_ticker}..."):
        df = fetch_data(selected_ticker)
        if df is None: st.error("Execution failure: Insufficient market flow data."); st.stop()
        
        # Calculate AI Components
        avg_sentiment, headlines, sentiment_scores = fetch_news_sentiment(selected_ticker)
        xgb_prob, shap_values, features = run_xgb_model(df, avg_sentiment, 3)
        sim_paths = run_monte_carlo(df, days=days_to_sim)
        
        # Generate SHAP dictionary
        shap_contrib = {feat: float(val) for feat, val in zip(features, shap_values.values[0])}
        
        # Fetch LLM Summary
        llm_advice = get_llm_summary(selected_ticker, df['Close'].iloc[-1], df['RSI'].iloc[-1], avg_sentiment, xgb_prob, shap_contrib, headlines)

        # Dashboard Metrics
        latest = df.iloc[-1]
        change = float(df['Returns'].iloc[-1]) * 100
        
        # 1. SLEEK METRIC CARDS
        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(f'<div class="metric-card"><div class="metric-label">Live Price</div><div class="metric-value">${latest["Close"]:.2f} ({change:+.2f}%)</div></div>', unsafe_allow_html=True)
        m2.markdown(f'<div class="metric-card"><div class="metric-label">RSI (14-Day)</div><div class="metric-value">{latest["RSI"]:.1f}</div></div>', unsafe_allow_html=True)
        m3.markdown(f'<div class="metric-card"><div class="metric-label">News Sentiment</div><div class="metric-value">{avg_sentiment:+.2f}</div></div>', unsafe_allow_html=True)
        m4.markdown(f'<div class="metric-card"><div class="metric-label">Neural Probability</div><div class="metric-value">{xgb_prob:.1f}%</div></div>', unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)

        # 2. INTERACTIVE GAUGE
        st.markdown("### 🎯 Final AI Decision Vector")
        st.markdown(f'<div class="gauge-container"><div class="gauge-bar" style="width: {xgb_prob}%;"></div></div>', unsafe_allow_html=True)
        
        banner_color = "#34D399" if xgb_prob >= 60.0 else ("#FBBF24" if xgb_prob > 40.0 else "#F87171")
        banner_text = "🟢 GO SIGNAL (BUY)" if xgb_prob >= 60.0 else ("⚖️ NEUTRAL (HOLD)" if xgb_prob > 40.0 else "🔴 NO-GO SIGNAL (SELL/AVOID)")
        st.markdown(f'<h3 style="text-align:center; color:{banner_color}; background:{banner_color}1A; padding:10px; border-radius:10px;">{banner_text} — {xgb_prob:.1f}% Prediction Confidence</h3>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # 3. ADVANCED ANALYTICS (TABS)
        tab_sum, tab_sim, tab_shap = st.tabs(["💡 AI Executive Advisor", "🔮 Monte Carlo Simulations", "🧬 Neural Explainability"])
        
        with tab_sum:
            col_l, col_r = st.columns([2, 1])
            with col_l:
                st.subheader("🤖 LLM Executive Advisor Bullets")
                st.markdown(llm_advice)
            with col_r:
                st.subheader("📰 Relevant Headlines")
                for h in headlines[:3]: st.caption(h); st.markdown("---")

        with tab_sim:
            st.subheader(f"🔮 Stochastic Monte Carlo Cone ({days_to_sim}-Day Price Flow)")
            sim_df = pd.DataFrame(sim_paths)
            
            p10 = sim_df.pct_change().iloc[1:].sum().quantile(0.10) * 100
            p50 = sim_df.pct_change().iloc[1:].sum().quantile(0.50) * 100
            p90 = sim_df.pct_change().iloc[1:].sum().quantile(0.90) * 100

            s1, s2, s3 = st.columns(3)
            s1.metric("Expected Growth (Median)", f"{p50:+.1f}%")
            s2.metric("Worst Case (10th Percentile)", f"{p10:+.1f}%")
            s3.metric("Best Case (90th Percentile)", f"{p90:+.1f}%")

            # Chart the cone
            st.line_chart(sim_paths[:, :15]) # Plot 15 sample paths

        with tab_shap:
            st.subheader("🧬 SHAP Value Neural Feature Attribution")
            st.write("This chart explains *why* the AI made the Go/No-Go decision by breaking down how much percentage contribution each feature provided to the final probability.")
            
            # Format SHAP data for charting
            shap_data = pd.DataFrame({
                'Feature': features,
                'Contribution': shap_values.values[0]
            })
            
            c = alt.Chart(shap_data).mark_bar().encode(
                x='Contribution:Q',
                y=alt.Y('Feature:N', sort='-x'),
                color=alt.condition(alt.datum.Contribution > 0, alt.value("#10B981"), alt.value("#EF4444"))
            ).properties(height=300)
            st.altair_chart(c, use_container_width=True)

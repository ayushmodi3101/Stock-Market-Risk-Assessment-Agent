import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from textblob import TextBlob
import socket
import requests

# Check Internet
def check_internet_connection(host="8.8.8.8", port=53, timeout=3):
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except socket.error:
        return False

# Load Past Performance
def load_performance_history():
    try:
        return pd.read_csv("performance_history.csv")
    except FileNotFoundError:
        return pd.DataFrame(columns=["Stock", "Return"])

# Learn From Past
def learn_from_past(stock, data):
    try:
        last_price = data['Close'].iloc[-2]
        new_price = data['Close'].iloc[-1]
        next_day_return = (new_price - last_price) / last_price

        history = load_performance_history()
        previous_returns = history[history['Stock'] == stock]
        if not previous_returns.empty:
            last_known_return = previous_returns['Return'].iloc[-1]
            change = next_day_return - last_known_return
            st.info(f"📅 Last seen: Return was {last_known_return:.2%}, now changed by {change:.2%}")
        else:
            st.info(f"📘 First time learning for {stock}: Return = {next_day_return:.2%}")

        updated = pd.concat([history, pd.DataFrame([{"Stock": stock, "Return": next_day_return}])], ignore_index=True)
        updated.to_csv("performance_history.csv", index=False)
    except Exception as e:
        st.error(f"Error during learning: {e}")

# Calculate Risk
def calculate_risk_metrics(stocks):
    results = []
    for stock in stocks:
        try:
            ticker = yf.Ticker(stock)
            data = ticker.history(period="1y")
            if data.empty:
                raise ValueError("No data returned.")
            data['Daily Return'] = data['Close'].pct_change()
            volatility = data['Daily Return'].std() * (252 ** 0.5)
            var_95 = np.percentile(data['Daily Return'].dropna(), 5)
            results.append({
                "Stock": stock,
                "Volatility": volatility,
                "VaR (95%)": var_95,
                "Data": data
            })
        except Exception as e:
            st.warning(f"Error fetching data for {stock}: {e}")
    return results

# Monitor Stock Price
def monitor_stock_price(stock):
    try:
        ticker = yf.Ticker(stock)
        data = ticker.history(period="1d", interval="1m")
        if data.empty:
            raise ValueError("No data returned.")
        return data['Close'].iloc[-1]
    except Exception as e:
        st.error(f"Error during real-time monitoring: {e}")
        return None

# News Headlines
def fetch_real_headlines(stock, api_key):
    try:
        search_term = stock.split('.')[0]
        url = f"https://newsapi.org/v2/everything?q={search_term}&sortBy=publishedAt&language=en&apiKey={api_key}"
        response = requests.get(url)
        data = response.json()
        if data["status"] != "ok" or not data["articles"]:
            st.warning(f"No recent news found for {search_term}.")
            return []
        return [article["title"] for article in data["articles"][:5]]
    except Exception as e:
        st.error(f"Error fetching news: {e}")
        return []

# Sentiment
def analyze_sentiment(stock, api_key):
    headlines = fetch_real_headlines(stock, api_key)
    if not headlines:
        return 0.0
    sentiments = [TextBlob(h).sentiment.polarity for h in headlines]
    avg_sentiment = np.mean(sentiments)
    st.subheader("📰 Latest Headlines")
    for h in headlines:
        st.markdown(f"- {h}")
    st.markdown(f"**🧠 Average Sentiment for {stock}: {avg_sentiment:.2f}**")
    return avg_sentiment

# Trend
def detect_trend(stock):
    try:
        data = yf.Ticker(stock).history(period="1y")
        if data.empty:
            return "unknown"
        data['MA50'] = data['Close'].rolling(window=50).mean()
        data['MA200'] = data['Close'].rolling(window=200).mean()
        if data['MA50'].iloc[-1] > data['MA200'].iloc[-1]:
            return "bullish"
        else:
            return "bearish"
    except Exception as e:
        st.warning(f"Error detecting trend: {e}")
        return "unknown"

# Visuals
def visualize_stock(stock, data):
    st.subheader("📊 Historical Price (1 Year)")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data['Close'], label=f"{stock} Close", color='blue')
    ax.set_title(f"{stock} Historical Prices")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)

    st.subheader("📈 Daily Return Distribution")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.hist(data['Daily Return'].dropna(), bins=50, alpha=0.75, color='orange')
    ax2.set_xlabel("Daily Return")
    ax2.set_ylabel("Frequency")
    st.pyplot(fig2)

# Main App
def main():
    st.set_page_config(page_title="Stock Risk Analyzer", page_icon="📈", layout="wide")
    st.title("📊 Adaptive Stock Market Risk Assessment Agent")
    st.info("📢 *This tool is for educational purposes only. Always consult a financial advisor.*")

    # Sidebar Inputs
    with st.sidebar:
        st.header("⚙️ Settings")
        api_key = st.text_input("🔑 NewsAPI Key", type="password")
        stock_input = st.text_input("📥 Enter Stock Tickers", value="TATAMOTORS.BO, INFY.BO")
        risk_tolerance = st.radio("🎯 Risk Tolerance", ["low", "medium", "high"])
        show_history = st.checkbox("📚 Show Learning History")
        use_custom_weights = st.checkbox("🧪 Use Custom Weights")
        if use_custom_weights:
            weight_vol = st.slider("Weight: Volatility", 0.0, 5.0, 1.0)
            weight_var = st.slider("Weight: VaR (95%)", 0.0, 5.0, 1.0)
            weight_ret = st.slider("Weight: Avg Past Return", -5.0, 5.0, 1.0)

    if show_history:
        st.subheader("📜 Learning History (Past Returns)")
        history_df = load_performance_history()
        if not history_df.empty:
            st.dataframe(history_df.tail(100))
            st.info(f"🗃️ Total records stored: {len(history_df)}")
            csv = history_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Learning History", data=csv, file_name="performance_history.csv", mime="text/csv")
        else:
            st.warning("📭 No past learning data available.")

    with st.expander("🔔 Real-Time Price Alert"):
        st.markdown("Set alert thresholds and enable live monitoring.")
        price_stock = st.text_input("📈 Stock Ticker for Monitoring", value="TATAMOTORS.BO")
        current_price = monitor_stock_price(price_stock)

        if current_price:
            st.metric(label=f"💵 {price_stock} Current Price", value=f"{current_price:.2f}")
            price_drop = st.number_input("🔻 Alert if price drops below:", value=float(current_price))
            price_rise = st.number_input("🔺 Alert if price rises above:", value=float(current_price * 1.05))
            check_interval = st.slider("⏱️ Check every (seconds)", 5, 60, 10)
            enable_monitoring = st.checkbox("🚨 Enable Real-Time Monitoring")

            if enable_monitoring:
                st.warning("📡 Live monitoring started. Keep this tab open.")
                placeholder = st.empty()
                alert_displayed = False
                max_checks = 100

                for i in range(max_checks):
                    latest_price = monitor_stock_price(price_stock)
                    if latest_price is None:
                        st.error("❌ Error fetching live price.")
                        break

                    placeholder.metric(label="📈 Live Price", value=f"{latest_price:.2f}")

                    if latest_price <= price_drop:
                        st.error(f"📉 Price fell below ₹{price_drop:.2f}! Current: ₹{latest_price:.2f}")
                        break
                    elif latest_price >= price_rise:
                        st.success(f"📈 Price rose above ₹{price_rise:.2f}! Current: ₹{latest_price:.2f}")
                        break

                    time.sleep(check_interval)

        else:
            st.warning("⚠️ Could not fetch current price for monitoring.")

    if not check_internet_connection():
        st.error("❌ No internet connection. Please reconnect and try again.")
        return

    if st.button("🔍 Analyze"):
        with st.spinner("Analyzing stocks... Please wait."):
            stocks = [s.strip().upper() for s in stock_input.split(",") if s.strip()]
            results = calculate_risk_metrics(stocks)

        if not results:
            st.warning("⚠️ No valid data retrieved for given stocks.")
            return

        results_df = pd.DataFrame([{
            "Stock": r["Stock"],
            "Volatility": r["Volatility"],
            "VaR (95%)": r["VaR (95%)"]
        } for r in results])

        performance_data = load_performance_history()
        if not performance_data.empty:
            avg_returns = performance_data.groupby("Stock")["Return"].mean().to_dict()
            results_df["Avg Past Return"] = results_df["Stock"].map(avg_returns).fillna(0)

            if use_custom_weights:
                results_df["Score"] = (
                    weight_vol * results_df["Volatility"] +
                    weight_var * results_df["VaR (95%)"].abs() -
                    weight_ret * results_df["Avg Past Return"]
                )
                st.info("🎯 Using custom weights for scoring.")
            else:
                if risk_tolerance == "low":
                    results_df["Score"] = (
                        2.0 * results_df["Volatility"] +
                        1.5 * results_df["VaR (95%)"].abs() -
                        1.0 * results_df["Avg Past Return"]
                    )
                elif risk_tolerance == "medium":
                    results_df["Score"] = (
                        1.0 * results_df["Volatility"] +
                        1.0 * results_df["VaR (95%)"].abs() -
                        1.0 * results_df["Avg Past Return"]
                    )
                else:
                    results_df["Score"] = (
                        0.5 * results_df["Volatility"] +
                        0.5 * results_df["VaR (95%)"].abs() -
                        1.5 * results_df["Avg Past Return"]
                    )
                st.info(f"📘 Applied default scoring for **{risk_tolerance}** risk tolerance.")
        else:
            results_df["Score"] = results_df["Volatility"] - results_df["VaR (95%)"]
            st.warning("⚠️ No historical performance data. Using default risk score.")

        if risk_tolerance == "high":
            best_stock_row = results_df.loc[results_df['Score'].idxmax()]
        else:
            best_stock_row = results_df.loc[results_df['Score'].idxmin()]
        best_stock = best_stock_row["Stock"]

        st.subheader("📈 Risk Metrics Table")
        st.dataframe(results_df.style.background_gradient(cmap='RdYlGn_r'))

        best_stock_data = next(r["Data"] for r in results if r["Stock"] == best_stock)

        st.success(f"🏆 Best Pick Based on Risk Score: **{best_stock}**")
        st.json(best_stock_row.to_dict())

        current_price = monitor_stock_price(best_stock)
        if current_price:
            st.metric(label=f"💵 {best_stock} Current Price", value=f"{current_price:.2f}")
            visualize_stock(best_stock, best_stock_data)

        if api_key:
            sentiment = analyze_sentiment(best_stock, api_key)
            st.progress((sentiment + 1) / 2)
            if sentiment < -0.1:
                st.error("⚠️ Negative market sentiment detected.")
            elif sentiment > 0.1:
                st.success("✅ Positive market sentiment detected.")
            else:
                st.info("🧭 Neutral sentiment observed.")

        trend = detect_trend(best_stock)
        if trend == "bullish":
            st.success(f"📈 {best_stock} is showing a **bullish trend**.")
        elif trend == "bearish":
            st.warning(f"📉 {best_stock} is showing a **bearish trend**.")
        else:
            st.info("🔍 Trend data unavailable.")

        learn_from_past(best_stock, best_stock_data)

if __name__ == "__main__":
    main()

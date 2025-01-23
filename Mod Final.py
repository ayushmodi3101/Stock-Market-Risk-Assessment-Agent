import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from textblob import TextBlob


# Function to calculate risk metrics
def calculate_risk_metrics(stocks):
    results = []
    for stock in stocks:
        try:
            # Fetch historical data
            ticker = yf.Ticker(stock)
            data = ticker.history(period="1y")
            
            # Calculate Daily Returns
            data['Daily Return'] = data['Close'].pct_change()
            
            # Calculate Risk Metrics
            volatility = data['Daily Return'].std() * (252 ** 0.5)  # Annualized volatility
            var_95 = np.percentile(data['Daily Return'].dropna(), 5)  # VaR
            
            # Store results
            results.append({
                "Stock": stock,
                "Volatility": volatility,
                "VaR (95%)": var_95,
                "Data": data  # Include historical data for plotting
            })
        except Exception as e:
            print(f"Error fetching data for {stock}: {e}")
    return results


# Function to monitor stock price in real-time
def monitor_stock_price(stock, price_threshold):
    print(f"\nMonitoring {stock} for price below {price_threshold}...")
    ticker = yf.Ticker(stock)
    while True:
        data = ticker.history(period="1d", interval="1m")
        current_price = data['Close'].iloc[-1]
        print(f"Current Price of {stock}: {current_price}")
        if current_price < price_threshold:
            print(f"ALERT: {stock} price dropped below {price_threshold}!")
            break
        time.sleep(60)  # Check every minute


# Function to analyze sentiment
def analyze_sentiment(stock):
    # Replace this with API integration to fetch real news
    headlines = [
        f"{stock} posts strong quarterly results",
        f"{stock} faces market volatility amid global tensions"
    ]
    sentiments = [TextBlob(h).sentiment.polarity for h in headlines]
    avg_sentiment = np.mean(sentiments)
    print(f"Average Sentiment for {stock}: {avg_sentiment:.2f}")
    return avg_sentiment


# Function to dynamically adjust risk score
def adaptive_risk_scoring(stock, sentiment_weight=0.2):
    metrics = calculate_risk_metrics([stock])[0]
    sentiment = analyze_sentiment(stock)
    dynamic_score = metrics["Volatility"] - metrics["VaR (95%)"] + sentiment_weight * sentiment
    print(f"Dynamic Risk Score for {stock}: {dynamic_score:.2f}")
    return dynamic_score


# Function to detect stock trends
def detect_trend(stock):
    data = yf.Ticker(stock).history(period="1y")
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['MA200'] = data['Close'].rolling(window=200).mean()
    if data['MA50'].iloc[-1] > data['MA200'].iloc[-1]:
        print(f"{stock} is in a bullish trend.")
        return "bullish"
    else:
        print(f"{stock} is in a bearish trend.")
        return "bearish"


# Function to visualize stock data
def visualize_stock(stock, data):
    plt.figure(figsize=(10, 6))
    plt.plot(data['Close'], label=f"{stock} Close Price")
    plt.title(f"{stock} Historical Prices (1 Year)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.hist(data['Daily Return'].dropna(), bins=50, alpha=0.75)
    plt.title(f"{stock} Daily Returns Distribution")
    plt.xlabel("Daily Return")
    plt.ylabel("Frequency")
    plt.show()


# Main program
def main():
    print("Welcome to the Reactive Stock Market Risk Assessment Agent!")
    
    # Step 1: User inputs
    num_stocks = int(input("How many stocks do you want to compare? "))
    stocks = []
    for _ in range(num_stocks):
        stock = input(f"Enter stock ticker (e.g., TATAMOTORS.BO): ").strip()
        stocks.append(stock)
    
    price_threshold = float(input("Enter price threshold for monitoring (e.g., 3500): "))
    user_risk_tolerance = input("Enter your risk tolerance (low, medium, high): ").strip().lower()

    # Step 2: Calculate risk metrics
    results = calculate_risk_metrics(stocks)
    results_df = pd.DataFrame([{
        "Stock": r["Stock"],
        "Volatility": r["Volatility"],
        "VaR (95%)": r["VaR (95%)"]
    } for r in results])
    results_df['Score'] = results_df['Volatility'] - results_df['VaR (95%)']  # Lower score is better

    # Step 3: Display results and best stock
    if not results_df.empty:
        print("\nRisk Metrics for Entered Stocks:")
        print(results_df)

        best_stock_row = results_df.loc[results_df['Score'].idxmin()]
        best_stock = best_stock_row["Stock"]
        best_stock_data = next(r["Data"] for r in results if r["Stock"] == best_stock)
        
        print("\nBest Stock Based on Risk Metrics:")
        print(best_stock_row)

        # Step 4: Real-time monitoring
        monitor_stock_price(best_stock, price_threshold)

        # Step 5: Sentiment analysis
        sentiment = analyze_sentiment(best_stock)
        if sentiment < -0.1:
            print("ALERT: Negative sentiment detected. Consider caution.")
        elif sentiment > 0.1:
            print("Positive sentiment detected. Consider buying opportunities.")

        # Step 6: Trend detection
        trend = detect_trend(best_stock)
        if trend == "bullish":
            print(f"{best_stock} is in a bullish trend. Consider buying.")
        elif trend == "bearish":
            print(f"{best_stock} is in a bearish trend. Consider caution.")

        # Step 7: Visualize the best stock's data
        visualize_stock(best_stock, best_stock_data)
    else:
        print("No valid stock data was processed.")

if __name__ == "__main__":
    main()


import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

# Main program
def main():
    # Step 1: User inputs
    num_stocks = int(input("How many stocks do you want to compare? "))
    stocks = []
    for _ in range(num_stocks):
        stock = input(f"Enter stock ticker (e.g., TATAMOTORS.BO): ").strip()
        stocks.append(stock)
    
    # Step 2: Calculate risk metrics
    results = calculate_risk_metrics(stocks)
    results_df = pd.DataFrame([{
        "Stock": r["Stock"],
        "Volatility": r["Volatility"],
        "VaR (95%)": r["VaR (95%)"]
    } for r in results])
    
    # Step 3: Add a composite score
    results_df['Score'] = results_df['Volatility'] - results_df['VaR (95%)']  # Lower score is better
    
    # Step 4: Select the best stock
    if not results_df.empty:
        best_stock_row = results_df.loc[results_df['Score'].idxmin()]
        best_stock = best_stock_row["Stock"]
        best_stock_data = next(r["Data"] for r in results if r["Stock"] == best_stock)
        
        print("\nRisk Metrics for Entered Stocks:")
        print(results_df)
        print("\nBest Stock Based on Risk Metrics:")
        print(best_stock_row)

        # Step 5: Plot the best stock's historical data
        plt.figure(figsize=(10, 6))
        plt.plot(best_stock_data['Close'], label=f"{best_stock} Close Price")
        plt.title(f"{best_stock} Historical Prices (1 Year)")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.show()

        # Plot Daily Returns Distribution
        plt.figure(figsize=(10, 6))
        plt.hist(best_stock_data['Daily Return'].dropna(), bins=50, alpha=0.75)
        plt.title(f"{best_stock} Daily Returns Distribution")
        plt.xlabel("Daily Return")
        plt.ylabel("Frequency")
        plt.show()
    else:
        print("No valid stock data was processed.")

# Run the program
if __name__ == "__main__":
    main()

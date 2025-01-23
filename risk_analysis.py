import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Fetch Data
bse = yf.Ticker("^BSESN")  # BSE Sensex ticker
market = yf.Ticker("^GSPC")  # S&P 500 as the market index

# Get 1-year historical data
bse_data = bse.history(period="1y")
market_data = market.history(period="1y")

# Calculate daily returns
bse_data['Daily Return'] = bse_data['Close'].pct_change()
market_data['Daily Return'] = market_data['Close'].pct_change()

# Step 2: Risk Metrics Calculation
# Volatility (Annualized)
volatility = bse_data['Daily Return'].std() * (252 ** 0.5)

# Beta (Sensitivity to Market Movements)
covariance = bse_data['Daily Return'].cov(market_data['Daily Return'])
market_variance = market_data['Daily Return'].var()
beta = covariance / market_variance

# Value at Risk (VaR) at 95% confidence level
var_95 = np.percentile(bse_data['Daily Return'].dropna(), 5)

# Print Risk Metrics
print(f"BSE Sensex Risk Metrics:")
print(f"Volatility (Annualized): {volatility:.2%}")
print(f"Beta: {beta:.2f}")
print(f"Value at Risk (95%): {var_95:.2%}")

# Step 3: Visualization
# Plot Historical Prices
plt.figure(figsize=(10, 6))
plt.plot(bse_data['Close'], label="BSE Sensex Close Price")
plt.title("BSE Sensex Historical Prices (1 Year)")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()

# Plot Daily Returns Distribution
plt.figure(figsize=(10, 6))
plt.hist(bse_data['Daily Return'].dropna(), bins=50, alpha=0.75)
plt.title("BSE Sensex Daily Returns Distribution")
plt.xlabel("Daily Return")
plt.ylabel("Frequency")
plt.show()

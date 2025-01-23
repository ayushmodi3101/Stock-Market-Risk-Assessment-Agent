import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Fetch Tata Motors data
tata_motors = yf.Ticker("TATAMOTORS.BO")  # Replace with your stock ticker
data = tata_motors.history(period="1y")

# Calculate daily returns
data['Daily Return'] = data['Close'].pct_change()

# Risk Metrics
volatility = data['Daily Return'].std() * (252 ** 0.5)  # Annualized volatility
var_95 = np.percentile(data['Daily Return'].dropna(), 5)  # VaR at 95% confidence

# Print Metrics
print(f"Tata Motors Risk Metrics:")
print(f"Volatility (Annualized): {volatility:.2%}")
print(f"Value at Risk (95%): {var_95:.2%}")

# Visualization
plt.figure(figsize=(10, 6))
plt.plot(data['Close'], label="Tata Motors Close Price")
plt.title("Tata Motors Historical Prices (1 Year)")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(data['Daily Return'].dropna(), bins=50, alpha=0.75)
plt.title("Tata Motors Daily Returns Distribution")
plt.xlabel("Daily Return")
plt.ylabel("Frequency")
plt.show()

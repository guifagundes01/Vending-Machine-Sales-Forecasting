import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import math

# Read the data
df = pd.read_csv('vending_machine_sales.csv')

# Convert date to datetime
df['TransDate'] = pd.to_datetime(df['TransDate'])

# Filter for Coca Cola Zero Sugar
coke_zero = df[df['Product'] == 'Coca Cola - Zero Sugar']

# Aggregate sales by month
monthly_quantity = coke_zero.groupby(pd.Grouper(key='TransDate', freq='ME'))['RQty'].sum()

# Calculate 3-month and 5-month moving averages
ma3 = monthly_quantity.rolling(window=3).mean()
ma5 = monthly_quantity.rolling(window=5).mean()

# Create the forecast for next month using both moving averages
last_ma3 = math.ceil(ma3.iloc[-1])
last_ma5 = math.ceil(ma5.iloc[-1])
forecast_date = monthly_quantity.index[-1] + pd.DateOffset(months=1)

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(monthly_quantity.index, monthly_quantity, label='Actual Quantity', marker='o')
plt.plot(ma3.index, ma3, label='3-Month Moving Average', linestyle='--')
plt.plot(ma5.index, ma5, label='5-Month Moving Average', linestyle=':')
plt.axvline(x=forecast_date, color='r', linestyle='--', label='Forecast Point')

# Add forecast points
plt.plot(forecast_date, last_ma3, 'ro', label=f'3M: {last_ma3}')
plt.plot(forecast_date, last_ma5, 'mo', label=f'5M: {last_ma5}')

plt.title('Coca Cola Zero Sugar - Monthly Quantity and Moving Averages')
plt.xlabel('Date')
plt.ylabel('Quantity Sold')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('output/moving_average/coke_zero_forecast.png')
plt.close()

# Print the results
print("\nCoca Cola Zero Sugar Quantity Analysis")
print("====================================")
print("\nMonthly Quantity:")
print(monthly_quantity)
print("\n3-Month Moving Average:")
print(ma3)
print("\n5-Month Moving Average:")
print(ma5)
print(f"\nForecast for next month:")
print(f"3-Month MA Forecast: {last_ma3} units")
print(f"5-Month MA Forecast: {last_ma5} units")

# Save results to a text file
with open('output/moving_average/coke_zero_forecast.txt', 'w') as f:
    f.write("Coca Cola Zero Sugar Quantity Forecast\n")
    f.write("====================================\n\n")
    f.write("Monthly Quantity:\n")
    f.write(monthly_quantity.to_string())
    f.write("\n\n3-Month Moving Average:\n")
    f.write(ma3.to_string())
    f.write("\n\n5-Month Moving Average:\n")
    f.write(ma5.to_string())
    f.write(f"\n\nForecast for next month:")
    f.write(f"\n3-Month MA Forecast: {last_ma3} units")
    f.write(f"\n5-Month MA Forecast: {last_ma5} units")

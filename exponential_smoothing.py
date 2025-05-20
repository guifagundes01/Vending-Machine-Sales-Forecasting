import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import math
from statsmodels.tsa.holtwinters import Holt as StatsHolt

def simple_exponential_smoothing(data, alpha):
    """Simple exponential smoothing"""
    forecast = [data[0]]  # Initialize with first value
    for i in range(1, len(data)):
        forecast.append(alpha * data[i-1] + (1 - alpha) * forecast[i-1])
    return forecast

def holts_method(data, alpha, beta):
    """Holt's method (trend-adjusted exponential smoothing)"""
    level = [data[0]]  # Initialize level
    trend = [data[1] - data[0]]  # Initialize trend
    forecast = [data[0]]  # Forecast for period 1 (t=0) is just the first value

    for i in range(1, len(data)):
        # Update level
        level.append(alpha * data[i] + (1 - alpha) * (level[i-1] + trend[i-1]))
        # Update trend
        trend.append(beta * (level[i] - level[i-1]) + (1 - beta) * trend[i-1])
        # Calculate forecast for period i (using info up to i-1)
        forecast.append(level[i-1] + trend[i-1])
        print(f"Period {i+1} level: {level[i]}")
        print(f"Period {i+1} trend: {trend[i]}")
        print(f"Period {i+1} forecast: {forecast[i]}")
    forecast.append(level[-1] + trend[-1])
    return forecast

# Read the data
df = pd.read_csv('vending_machine_sales.csv')

# Convert date to datetime
df['TransDate'] = pd.to_datetime(df['TransDate'])

# Filter for Coca Cola Zero Sugar
coke_zero = df[df['Product'] == 'Coca Cola - Zero Sugar']

# Aggregate sales by month
monthly_quantity = coke_zero.groupby(pd.Grouper(key='TransDate', freq='ME'))['RQty'].sum()

# Convert to list for calculations
data = monthly_quantity.values.tolist()

# Calculate forecasts using different methods
# a. Simple exponential smoothing with alpha = 0.3
forecast_alpha03 = simple_exponential_smoothing(data, 0.3)
next_forecast_alpha03 = math.ceil(0.3 * data[-1] + (1 - 0.3) * forecast_alpha03[-1])

# b. Simple exponential smoothing with alpha = 0.5
forecast_alpha05 = simple_exponential_smoothing(data, 0.5)
next_forecast_alpha05 = math.ceil(0.5 * data[-1] + (1 - 0.5) * forecast_alpha05[-1])

# c. Holt's method with alpha = 0.5 and beta = 0.3
forecast_holts = holts_method(data, 0.5, 0.3)
next_forecast_holts = math.ceil(forecast_holts[-1])
forecast_holts.pop()

# # Compare with statsmodels implementation
# stats_holt = StatsHolt(data)
# stats_model = stats_holt.fit(smoothing_level=0.5, smoothing_trend=0.3)
# stats_forecast = stats_model.fittedvalues
# stats_next_forecast = stats_model.forecast(1)[0]

# print("\nComparison between our implementation and statsmodels:")
# print("==================================================")
# print("\nOur implementation:")
# for i, (date, forecast) in enumerate(zip(monthly_quantity.index, forecast_holts)):
#     print(f"Period {i+1} ({date.strftime('%Y-%m')}): {math.ceil(forecast)} units")

# print("\nStatsmodels implementation:")
# for i, (date, forecast) in enumerate(zip(monthly_quantity.index, stats_forecast)):
#     print(f"Period {i+1} ({date.strftime('%Y-%m')}): {math.ceil(forecast)} units")

# print("\nNext period forecast comparison:")
# print(f"Our implementation: {next_forecast_holts} units")
# print(f"Statsmodels: {math.ceil(stats_next_forecast)} units")

# Create the plot
plt.figure(figsize=(15, 8))
plt.plot(monthly_quantity.index, data, 'ko-', label='Actual Quantity', marker='o')
plt.plot(monthly_quantity.index, forecast_alpha03, 'r--', label='Exp Smoothing (alpha=0.3)')
plt.plot(monthly_quantity.index, forecast_alpha05, 'b--', label='Exp Smoothing (alpha=0.5)')
plt.plot(monthly_quantity.index, forecast_holts, 'g--', label="Our Holt's Method (alpha=0.5, beta=0.3)")
# plt.plot(monthly_quantity.index, stats_forecast, 'y--', label="Statsmodels Holt's Method (alpha=0.5, beta=0.3)")

# Add forecast points
forecast_date = monthly_quantity.index[-1] + pd.DateOffset(months=1)
plt.plot(forecast_date, next_forecast_alpha03, 'ro', label=f'Forecast alpha=0.3: {next_forecast_alpha03}')
plt.plot(forecast_date, next_forecast_alpha05, 'bo', label=f'Forecast alpha=0.5: {next_forecast_alpha05}')
plt.plot(forecast_date, next_forecast_holts, 'go', label=f"Our Forecast Holt's: {next_forecast_holts}")
# plt.plot(forecast_date, stats_next_forecast, 'yo', label=f"Statsmodels Forecast: {math.ceil(stats_next_forecast)}")

plt.title('Coca Cola Zero Sugar - Exponential Smoothing Forecasts')
plt.xlabel('Date')
plt.ylabel('Quantity')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('output/exponential_smoothing/exponential_smoothing_forecast.png')
plt.close()

# Print results
print("\nCoca Cola Zero Sugar - Exponential Smoothing Analysis")
print("==================================================")
print("\nMonthly Quantity:")
print(monthly_quantity)

print("\nMonthly Forecasts:")
print("\nSimple Exponential Smoothing (alpha=0.3):")
for i, (date, forecast) in enumerate(zip(monthly_quantity.index, forecast_alpha03)):
    print(f"Period {i+1} ({date.strftime('%Y-%m')}): {math.ceil(forecast)} units")

print("\nSimple Exponential Smoothing (alpha=0.5):")
for i, (date, forecast) in enumerate(zip(monthly_quantity.index, forecast_alpha05)):
    print(f"Period {i+1} ({date.strftime('%Y-%m')}): {math.ceil(forecast)} units")

print("\nHolt's Method (alpha=0.5, beta=0.3):")
for i, (date, forecast) in enumerate(zip(monthly_quantity.index, forecast_holts)):
    print(f"Period {i+1} ({date.strftime('%Y-%m')}): {math.ceil(forecast)} units")

print(f"\nForecast for next month (period 13):")
print(f"Simple Exponential Smoothing (alpha=0.3): {next_forecast_alpha03} units")
print(f"Simple Exponential Smoothing (alpha=0.5): {next_forecast_alpha05} units")
print(f"Holt's Method (alpha=0.5, beta=0.3): {next_forecast_holts} units")

# Save results to a text file
with open('output/exponential_smoothing/exponential_smoothing_forecast.txt', 'w') as f:
    f.write("Coca Cola Zero Sugar - Exponential Smoothing Analysis\n")
    f.write("==================================================\n\n")
    f.write("Monthly Quantity:\n")
    f.write(monthly_quantity.to_string())
    
    f.write("\n\nMonthly Forecasts:\n")
    f.write("\nSimple Exponential Smoothing (alpha=0.3):\n")
    for i, (date, forecast) in enumerate(zip(monthly_quantity.index, forecast_alpha03)):
        f.write(f"Period {i+1} ({date.strftime('%Y-%m')}): {math.ceil(forecast)} units\n")
    
    f.write("\nSimple Exponential Smoothing (alpha=0.5):\n")
    for i, (date, forecast) in enumerate(zip(monthly_quantity.index, forecast_alpha05)):
        f.write(f"Period {i+1} ({date.strftime('%Y-%m')}): {math.ceil(forecast)} units\n")
    
    f.write("\nHolt's Method (alpha=0.5, beta=0.3):\n")
    for i, (date, forecast) in enumerate(zip(monthly_quantity.index, forecast_holts)):
        f.write(f"Period {i+1} ({date.strftime('%Y-%m')}): {math.ceil(forecast)} units\n")
    
    f.write(f"\n\nForecast for next month (period 13):")
    f.write(f"\nSimple Exponential Smoothing (alpha=0.3): {next_forecast_alpha03} units")
    f.write(f"\nSimple Exponential Smoothing (alpha=0.5): {next_forecast_alpha05} units")
    f.write(f"\nHolt's Method (alpha=0.5, beta=0.3): {next_forecast_holts} units")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import os

from datetime import datetime

def simple_exponential_smoothing(data, alpha):
    forecast = [data[0]]
    for i in range(1, len(data)):
        forecast.append(alpha * data[i-1] + (1 - alpha) * forecast[i-1])
    return forecast

def holts_method(data, alpha, beta):
    level = [data[0]]
    trend = [data[1] - data[0]]
    forecast = [data[0]]
    for i in range(1, len(data)):
        level.append(alpha * data[i] + (1 - alpha) * (level[i-1] + trend[i-1]))
        trend.append(beta * (level[i] - level[i-1]) + (1 - beta) * trend[i-1])
        forecast.append(level[i-1] + trend[i-1])
    forecast.append(level[-1] + trend[-1])
    return forecast

# Read the data
os.makedirs('output/exponential_smoothing/top10', exist_ok=True)
df = pd.read_csv('vending_machine_sales.csv')
df['TransDate'] = pd.to_datetime(df['TransDate'])

# Get top 10 products by total quantity
top_products = df.groupby('Product')['RQty'].sum().sort_values(ascending=False).head(10).index

for product in top_products:
    product_data = df[df['Product'] == product]
    monthly_quantity = product_data.groupby(pd.Grouper(key='TransDate', freq='ME'))['RQty'].sum()
    data = monthly_quantity.values.tolist()
    if len(data) < 2:
        continue  # Not enough data for Holt's method

    # Simple exponential smoothing
    forecast_alpha03 = simple_exponential_smoothing(data, 0.3)
    next_forecast_alpha03 = math.ceil(0.3 * data[-1] + (1 - 0.3) * forecast_alpha03[-1])
    forecast_alpha05 = simple_exponential_smoothing(data, 0.5)
    next_forecast_alpha05 = math.ceil(0.5 * data[-1] + (1 - 0.5) * forecast_alpha05[-1])

    # Holt's method
    forecast_holts = holts_method(data, 0.5, 0.3)
    next_forecast_holts = math.ceil(forecast_holts[-1])
    forecast_holts = forecast_holts[:-1]

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(monthly_quantity.index, data, 'ko-', label='Actual Quantity', marker='o')
    plt.plot(monthly_quantity.index, forecast_alpha03, 'r--', label='Exp Smoothing (alpha=0.3)')
    plt.plot(monthly_quantity.index, forecast_alpha05, 'b--', label='Exp Smoothing (alpha=0.5)')
    plt.plot(monthly_quantity.index, forecast_holts, 'g--', label="Holt's Method (alpha=0.5, beta=0.3)")
    forecast_date = monthly_quantity.index[-1] + pd.DateOffset(months=1)
    plt.plot(forecast_date, next_forecast_alpha03, 'ro', label=f'Forecast alpha=0.3: {next_forecast_alpha03}')
    plt.plot(forecast_date, next_forecast_alpha05, 'bo', label=f'Forecast alpha=0.5: {next_forecast_alpha05}')
    plt.plot(forecast_date, next_forecast_holts, 'go', label=f"Forecast Holt's: {next_forecast_holts}")
    plt.title(f'{product} - Exponential Smoothing Forecasts')
    plt.xlabel('Date')
    plt.ylabel('Quantity')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    safe_product = product.replace('/', '_').replace(' ', '_')
    plt.savefig(f'output/exponential_smoothing/top10/{safe_product}_exponential_smoothing.png')
    plt.close()

    # Print and save results
    with open(f'output/exponential_smoothing/top10/{safe_product}_exponential_smoothing.txt', 'w') as f:
        f.write(f"{product} - Exponential Smoothing Analysis\n")
        f.write("=" * (len(product) + 30) + "\n\n")
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
        f.write(f"\n\nForecast for next month (period {len(data)+1}):\n")
        f.write(f"Simple Exponential Smoothing (alpha=0.3): {next_forecast_alpha03} units\n")
        f.write(f"Simple Exponential Smoothing (alpha=0.5): {next_forecast_alpha05} units\n")
        f.write(f"Holt's Method (alpha=0.5, beta=0.3): {next_forecast_holts} units\n") 
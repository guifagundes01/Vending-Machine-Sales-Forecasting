import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import math

# Create output directory if it doesn't exist
os.makedirs('output/moving_average', exist_ok=True)

# Read the data
df = pd.read_csv('vending_machine_sales.csv')

# Convert date to datetime
df['TransDate'] = pd.to_datetime(df['TransDate'])

# Get top 10 products by total quantity
top_products = df.groupby('Product')['RQty'].sum().sort_values(ascending=False).head(10).index

# Create a figure for all products
plt.figure(figsize=(15, 10))

# Process each top product
for idx, product in enumerate(top_products, 1):
    # Filter for current product
    product_data = df[df['Product'] == product]
    
    # Aggregate sales by month
    monthly_quantity = product_data.groupby(pd.Grouper(key='TransDate', freq='ME'))['RQty'].sum()
    
    # Calculate 3-month and 5-month moving averages
    ma3 = monthly_quantity.rolling(window=3).mean()
    ma5 = monthly_quantity.rolling(window=5).mean()
    
    # Create the forecast for next month using both moving averages
    last_ma3 = math.ceil(ma3.iloc[-1])
    last_ma5 = math.ceil(ma5.iloc[-1])
    forecast_date = monthly_quantity.index[-1] + pd.DateOffset(months=1)
    
    # Create subplot for each product
    plt.subplot(5, 2, idx)
    plt.plot(monthly_quantity.index, monthly_quantity, label='Actual Quantity', marker='o', markersize=3)
    plt.plot(ma3.index, ma3, label='3-Month MA', linestyle='--')
    plt.plot(ma5.index, ma5, label='5-Month MA', linestyle=':')
    plt.axvline(x=forecast_date, color='r', linestyle='--', label='Forecast Point')
    
    # Add forecast points
    plt.plot(forecast_date, last_ma3, 'ro', label=f'3M: {last_ma3}')
    plt.plot(forecast_date, last_ma5, 'mo', label=f'5M: {last_ma5}')
    
    plt.title(f'{product[:30]}...' if len(product) > 30 else product)
    plt.xlabel('Date')
    plt.ylabel('Quantity')
    plt.legend(fontsize='small')
    plt.grid(True)
    plt.xticks(rotation=45)
    
    # Save individual product results
    with open(f'output/moving_average/{product.replace("/", "_")}_forecast.txt', 'w') as f:
        f.write(f"{product} Quantity Forecast\n")
        f.write("=" * (len(product) + 20) + "\n\n")
        f.write("Monthly Quantity:\n")
        f.write(monthly_quantity.to_string())
        f.write("\n\n3-Month Moving Average:\n")
        f.write(ma3.to_string())
        f.write("\n\n5-Month Moving Average:\n")
        f.write(ma5.to_string())
        f.write(f"\n\nForecast for next month:")
        f.write(f"\n3-Month MA Forecast: {last_ma3} units")
        f.write(f"\n5-Month MA Forecast: {last_ma5} units")

plt.tight_layout()
plt.savefig('output/moving_average/top10_products_forecast.png')
plt.close()

# Create summary file
with open('output/moving_average/top10_forecast_summary.txt', 'w') as f:
    f.write("Top 10 Products Forecast Summary\n")
    f.write("==============================\n\n")
    
    for product in top_products:
        product_data = df[df['Product'] == product]
        monthly_quantity = product_data.groupby(pd.Grouper(key='TransDate', freq='ME'))['RQty'].sum()
        ma3 = monthly_quantity.rolling(window=3).mean()
        ma5 = monthly_quantity.rolling(window=5).mean()
        
        f.write(f"\nProduct: {product}\n")
        f.write("-" * (len(product) + 10) + "\n")
        f.write(f"Total Quantity Sold: {product_data['RQty'].sum():,.0f}\n")
        f.write(f"3-Month MA Forecast: {math.ceil(ma3.iloc[-1]):,.0f} units\n")
        f.write(f"5-Month MA Forecast: {math.ceil(ma5.iloc[-1]):,.0f} units\n") 
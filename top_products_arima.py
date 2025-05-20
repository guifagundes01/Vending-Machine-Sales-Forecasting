import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats
import warnings
import math
import os
warnings.filterwarnings('ignore')

def check_stationarity(timeseries):
    """Check if the time series is stationary using Augmented Dickey-Fuller test"""
    result = adfuller(timeseries)
    return result[1] < 0.05  # Return True if stationary

def apply_differencing(data, order=1):
    """Apply differencing to the time series"""
    diff_data = np.diff(data, n=order)
    return diff_data

def plot_forecast(data, dates, forecast, forecast_date, product_name):
    """Plot the forecast results"""
    plt.figure(figsize=(15, 8))
    plt.plot(dates, data, 'ko-', label='Actual Quantity', marker='o')
    plt.plot(dates[1:], forecast[1:], 'r--', label='ARIMA Fitted Values')
    plt.plot(forecast_date, forecast[-1], 'ro', label=f'Forecast: {math.ceil(forecast[-1])}')
    
    plt.title(f'{product_name} - ARIMA Forecast')
    plt.xlabel('Date')
    plt.ylabel('Quantity')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'output/arima/top_products/forecast_{product_name.replace("/", "_")}.png')
    plt.close()

def perform_arima_analysis(product_data, product_name):
    """Perform complete ARIMA analysis for a product"""
    # Convert to list for calculations
    data = product_data.values.tolist()
    dates = product_data.index
    
    # Check stationarity
    is_stationary = check_stationarity(data)
    
    # Apply differencing
    diff_data = apply_differencing(data)
    
    # Check stationarity of differenced data
    is_diff_stationary = check_stationarity(diff_data)
    
    # Fit ARIMA model
    model = ARIMA(data, order=(1, 1, 1))
    model_fit = model.fit()
    
    # Get residuals and forecast
    residuals = model_fit.resid
    forecast = model_fit.fittedvalues
    next_forecast = model_fit.forecast(steps=1)[0]
    
    # Plot forecast
    forecast_date = dates[-1] + pd.DateOffset(months=1)
    plot_forecast(data, dates, forecast, forecast_date, product_name)
    
    # Calculate statistics
    stats_dict = {
        'Mean of residuals': np.mean(residuals),
        'Std of residuals': np.std(residuals),
        'Skewness': stats.skew(residuals),
        'Kurtosis': stats.kurtosis(residuals),
        'Next month forecast': math.ceil(next_forecast)
    }
    
    return stats_dict

def main():
    # Create output directory if it doesn't exist
    os.makedirs('output/arima/top_products', exist_ok=True)
    
    # Read the data
    df = pd.read_csv('vending_machine_sales.csv')
    df['TransDate'] = pd.to_datetime(df['TransDate'])
    
    # Get top 10 products by total quantity
    top_products = df.groupby('Product')['RQty'].sum().nlargest(10)
    
    # Initialize results dictionary
    results = {}
    
    # Perform analysis for each top product
    for product in top_products.index:
        print(f"\nAnalyzing {product}...")
        
        # Filter and aggregate data for the product
        product_data = df[df['Product'] == product].groupby(pd.Grouper(key='TransDate', freq='ME'))['RQty'].sum()
        
        # Perform ARIMA analysis
        stats_dict = perform_arima_analysis(product_data, product)
        results[product] = stats_dict
        
        # Print results
        print(f"\nResults for {product}:")
        print("==================")
        for stat, value in stats_dict.items():
            print(f"{stat}: {value:.4f}")
    
    # Save results to a text file
    with open('output/arima/top_products/analysis_results.txt', 'w') as f:
        f.write("ARIMA Analysis Results for Top 10 Products\n")
        f.write("=========================================\n\n")
        
        for product, stats in results.items():
            f.write(f"\n{product}\n")
            f.write("-" * len(product) + "\n")
            for stat, value in stats.items():
                f.write(f"{stat}: {value:.4f}\n")

if __name__ == "__main__":
    main() 
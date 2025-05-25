import pandas as pd
import numpy as np
import math
from statsmodels.tsa.holtwinters import Holt as StatsHolt
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

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

def moving_average_forecast(data, window):
    ma = pd.Series(data).rolling(window=window).mean()
    return ma.iloc[-1], ma

def mse(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean((y_true - y_pred) ** 2)

def arima_forecast_and_mse(data, product_name=None):
    data = np.array(data, dtype=float)
    if len(data) < 6 or np.count_nonzero(data) < 3 or np.std(data) == 0:
        if product_name:
            print(f"ARIMA skipped for {product_name}: insufficient or constant data.")
        return np.nan, np.nan, np.nan
    try:
        model = ARIMA(data, order=(1, 1, 1))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=2)  # Forecast 2 months ahead
        fitted = model_fit.fittedvalues
        # Explicitly skip the first fitted value
        y_true = data[1:]
        y_pred = fitted[1:]
        mse_val = mse(y_true, y_pred)
        return math.ceil(forecast[0]), math.ceil(forecast[1]), mse_val
    except Exception as e:
        if product_name:
            print(f'ARIMA failed for {product_name}: {e}')
        return np.nan, np.nan, np.nan

def main():
    df = pd.read_csv('vending_machine_sales.csv')
    df['TransDate'] = pd.to_datetime(df['TransDate'])
    
    # Get total quantity per product
    product_totals = df.groupby('Product')['RQty'].sum().sort_values(ascending=False)
    products = product_totals.index.tolist()
    
    results = []
    for product in products:
        prod_df = df[df['Product'] == product].copy()
        monthly = prod_df.groupby(pd.Grouper(key='TransDate', freq='ME'))['RQty'].sum()
        # Fill missing months with 0
        all_months = pd.date_range(start=monthly.index.min(), end=monthly.index.max(), freq='ME')
        monthly = monthly.reindex(all_months, fill_value=0)
        data = monthly.values.tolist()
        # Skip products with less than 3 months of data
        if len(data) < 3:
            continue
        # Moving averages
        ma3_val, ma3_series = moving_average_forecast(data, 3)
        ma5_val, ma5_series = moving_average_forecast(data, 5)
        # MA3 MSE (start at index 2, compare data[3:] with ma3_series[2:-1])
        ma3_start = 2
        ma3_mse = mse(data[ma3_start+1:], ma3_series[ma3_start:-1]) if len(data) > ma3_start+1 else ''
        # MA5 MSE (start at index 4, compare data[5:] with ma5_series[4:-1])
        ma5_start = 4
        ma5_mse = mse(data[ma5_start+1:], ma5_series[ma5_start:-1]) if len(data) > ma5_start+1 else ''
        # SES
        ses03 = simple_exponential_smoothing(data, 0.3)
        next_ses03 = math.ceil(0.3 * data[-1] + (1 - 0.3) * ses03[-1])
        next_ses03_2 = math.ceil(0.3 * next_ses03 + (1 - 0.3) * ses03[-1])
        ses03_mse = mse(data[1:], ses03[1:])
        ses05 = simple_exponential_smoothing(data, 0.5)
        next_ses05 = math.ceil(0.5 * data[-1] + (1 - 0.5) * ses05[-1])
        next_ses05_2 = math.ceil(0.5 * next_ses05 + (1 - 0.5) * ses05[-1])
        ses05_mse = mse(data[1:], ses05[1:])
        # Holt's method
        holt = holts_method(data, 0.5, 0.3)
        next_holt = math.ceil(holt[-1])
        next_holt_2 = math.ceil(holt[-1] + (holt[-1] - holt[-2]))
        holt_mse = mse(data[1:], holt[1:-1])
        # ARIMA
        next_arima, next_arima_2, arima_mse = arima_forecast_and_mse(data, product)
        results.append({
            'Product': product,
            'TotalQuantity': int(product_totals[product]),
            'MA3': int(ma3_val) if not np.isnan(ma3_val) else '',
            'MA3_MSE': ma3_mse if ma3_mse != '' else '',
            'MA5': int(ma5_val) if not np.isnan(ma5_val) else '',
            'MA5_MSE': ma5_mse if ma5_mse != '' else '',
            'SES_0.3_M1': next_ses03,
            'SES_0.3_M2': next_ses03_2,
            'SES_0.3_MSE': ses03_mse,
            'SES_0.5_M1': next_ses05,
            'SES_0.5_M2': next_ses05_2,
            'SES_0.5_MSE': ses05_mse,
            'Holt_M1': next_holt,
            'Holt_M2': next_holt_2,
            'Holt_MSE': holt_mse,
            'ARIMA_M1': next_arima if not np.isnan(next_arima) else '',
            'ARIMA_M2': next_arima_2 if not np.isnan(next_arima_2) else '',
            'ARIMA_MSE': arima_mse if not np.isnan(arima_mse) else ''
        })
    # Sort by TotalQuantity
    results = sorted(results, key=lambda x: x['TotalQuantity'], reverse=True)
    # Save to CSV
    out_df = pd.DataFrame(results)
    out_df.to_csv('output/all_methods_forecast_summary.csv', index=False)
    print('Forecast summary saved to output/all_methods_forecast_summary.csv')

if __name__ == '__main__':
    main() 
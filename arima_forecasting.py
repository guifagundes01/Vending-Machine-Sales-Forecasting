import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
import math
from scipy import stats
warnings.filterwarnings('ignore')

def check_stationarity(timeseries):
    """Check if the time series is stationary using Augmented Dickey-Fuller test"""
    result = adfuller(timeseries)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical values:', result[4])
    return result[1] < 0.05  # Return True if stationary

def apply_differencing(data, order=1):
    """Apply differencing to the time series"""
    diff_data = np.diff(data, n=order)
    return diff_data

def plot_original_and_differenced(original_data, diff_data, dates):
    """Plot original and differenced time series"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Plot original data
    ax1.plot(dates, original_data, 'ko-', label='Original Data', marker='o')
    ax1.set_title('Original Time Series')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Quantity')
    ax1.legend()
    ax1.grid(True)
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot differenced data
    ax2.plot(dates[1:], diff_data, 'ro-', label='First Difference', marker='o')
    ax2.set_title('First Difference of Time Series')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Difference')
    ax2.legend()
    ax2.grid(True)
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('output/arima/differencing_plots.png')
    plt.close()

def plot_acf_pacf(data, lags=None):
    """Plot ACF and PACF to help determine ARIMA parameters"""
    max_lags = max(1, (len(data) // 2) - 1)
    if lags is None or lags > max_lags:
        lags = max_lags
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    plot_acf(data, lags=lags, ax=ax1)
    plot_pacf(data, lags=lags, ax=ax2)
    plt.tight_layout()
    plt.savefig('output/arima/acf_pacf_plots.png')
    plt.close()

def plot_residuals_analysis(residuals, dates):
    """Plot residual analysis including time series, histogram, and Q-Q plot"""
    fig = plt.figure(figsize=(15, 10))
    
    # Time series of residuals
    ax1 = plt.subplot(311)
    ax1.plot(dates, residuals, 'ko-', marker='o')
    ax1.axhline(y=0, color='r', linestyle='--')
    ax1.set_title('Residuals Time Series')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Residuals')
    ax1.grid(True)
    ax1.tick_params(axis='x', rotation=45)
    
    # Histogram of residuals
    ax2 = plt.subplot(312)
    ax2.hist(residuals, bins=20, density=True, alpha=0.7, color='blue')
    ax2.set_title('Histogram of Residuals')
    ax2.set_xlabel('Residual Value')
    ax2.set_ylabel('Density')
    
    # Add normal distribution curve
    xmin, xmax = ax2.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, np.mean(residuals), np.std(residuals))
    ax2.plot(x, p, 'k', linewidth=2)
    
    # Q-Q plot
    ax3 = plt.subplot(313)
    stats.probplot(residuals, dist="norm", plot=ax3)
    ax3.set_title('Q-Q Plot of Residuals')
    
    plt.tight_layout()
    plt.savefig('output/arima/residuals_analysis.png')
    plt.close()

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
dates = monthly_quantity.index

# Check stationarity of original data
print("\nChecking stationarity of the original time series:")
is_stationary = check_stationarity(data)
print(f"The original time series is {'stationary' if is_stationary else 'not stationary'}")

# Apply differencing
diff_data = apply_differencing(data)

# Check stationarity of differenced data
print("\nChecking stationarity of the differenced time series:")
is_diff_stationary = check_stationarity(diff_data)
print(f"The differenced time series is {'stationary' if is_diff_stationary else 'not stationary'}")

# Plot original and differenced series
plot_original_and_differenced(data, diff_data, dates)

# Plot ACF and PACF of differenced data
plot_acf_pacf(diff_data)

# Fit ARIMA model with differencing
model = ARIMA(data, order=(1, 1, 1))  # d=1 for first difference
model_fit = model.fit()

# Get residuals
residuals = model_fit.resid

# Plot residuals analysis
plot_residuals_analysis(residuals, dates)

# Print residual statistics
print("\nResidual Analysis:")
print("==================")
print(f"Mean of residuals: {np.mean(residuals):.4f}")
print(f"Standard deviation of residuals: {np.std(residuals):.4f}")
print(f"Skewness of residuals: {stats.skew(residuals):.4f}")
print(f"Kurtosis of residuals: {stats.kurtosis(residuals):.4f}")

# Calculate MSE
mse = np.mean((data[1:] - model_fit.fittedvalues[1:]) ** 2)
print(f"\nMean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {np.sqrt(mse):.4f}")

# Perform Ljung-Box test for residual autocorrelation
from statsmodels.stats.diagnostic import acorr_ljungbox
lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
print("\nLjung-Box Test for Residual Autocorrelation:")
print(lb_test)

# Print model summary
print("\nARIMA Model Summary:")
print(model_fit.summary())

# Make forecast for next month
forecast = model_fit.forecast(steps=1)
next_forecast = math.ceil(forecast[0])

# Plot the results
plt.figure(figsize=(15, 8))
plt.plot(dates, data, 'ko-', label='Actual Quantity', marker='o')
plt.plot(dates[1:], model_fit.fittedvalues[1:], 'r--', label='ARIMA Fitted Values')

# Add forecast point
forecast_date = dates[-1] + pd.DateOffset(months=1)
plt.plot(forecast_date, next_forecast, 'ro', label=f'ARIMA Forecast: {next_forecast}')

plt.title('Coca Cola Zero Sugar - ARIMA Forecast')
plt.xlabel('Date')
plt.ylabel('Quantity')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('output/arima/arima_forecast.png')
plt.close()

# Print results
print("\nCoca Cola Zero Sugar - ARIMA Analysis")
print("==================================================")
print("\nMonthly Quantity:")
print(monthly_quantity)

print("\nMonthly Fitted Values:")
for i, (date, fitted) in enumerate(zip(dates[1:], model_fit.fittedvalues[1:])):
    print(f"Period {i+2} ({date.strftime('%Y-%m')}): {math.ceil(fitted)} units")

print(f"\nForecast for next month (period {len(dates)+1}):")
print(f"ARIMA(1,1,1) Forecast: {next_forecast} units")

# Save results to a text file
with open('output/arima/arima_forecast.txt', 'w') as f:
    f.write("Coca Cola Zero Sugar - ARIMA Analysis\n")
    f.write("==================================================\n\n")
    f.write("Monthly Quantity:\n")
    f.write(monthly_quantity.to_string())
    
    f.write("\n\nMonthly Fitted Values:\n")
    for i, (date, fitted) in enumerate(zip(dates[1:], model_fit.fittedvalues[1:])):
        f.write(f"Period {i+2} ({date.strftime('%Y-%m')}): {math.ceil(fitted)} units\n")
    
    f.write(f"\n\nForecast for next month (period {len(dates)+1}):")
    f.write(f"\nARIMA(1,1,1) Forecast: {next_forecast} units") 
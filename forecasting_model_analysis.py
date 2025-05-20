import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the summary CSV
df = pd.read_csv('output/all_methods_forecast_summary.csv')

# List of MSE columns
mse_cols = ['MA3_MSE', 'MA5_MSE', 'SES_0.3_MSE', 'SES_0.5_MSE', 'Holt_MSE', 'ARIMA_MSE']

# Ensure output directory exists
os.makedirs('output', exist_ok=True)

# Find the method with the lowest MSE for each product
df['BestMethod'] = df[mse_cols].astype(float).idxmin(axis=1)

# Count how often each method is best
best_counts = df['BestMethod'].value_counts()

# Plot: Best method count
plt.figure(figsize=(10,6))
best_counts.plot(kind='bar', color='skyblue')
plt.title('Best Forecasting Method by Product (Lowest MSE)')
plt.xlabel('Forecasting Method')
plt.ylabel('Number of Products (Best MSE)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('output/best_method_barplot.png')
plt.close()

# Plot: Average MSE per method
avg_mse = df[mse_cols].astype(float).mean()
plt.figure(figsize=(10,6))
avg_mse.plot(kind='bar', color='orange')
plt.title('Average MSE by Forecasting Method')
plt.ylabel('Average MSE')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('output/average_mse_barplot.png')
plt.close()

print('Analysis complete. Plots saved to output/.') 
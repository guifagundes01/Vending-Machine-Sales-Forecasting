import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Read the CSV file
df = pd.read_csv('vending_machine_sales.csv')

# Convert date columns to datetime
df['TransDate'] = pd.to_datetime(df['TransDate'])
df['Prcd Date'] = pd.to_datetime(df['Prcd Date'])

# 1. Sales by Location
location_sales = df.groupby('Location')['LineTotal'].sum().sort_values(ascending=False)
print("\nTotal Sales by Location:")
print(location_sales)

# 2. Most Popular Products
product_sales = df.groupby('Product')['LineTotal'].sum().sort_values(ascending=False).head(10)
print("\nTop 10 Products by Revenue:")
print(product_sales)

# 3. Revenue Analysis
category_sales = df.groupby('Category')['LineTotal'].sum().sort_values(ascending=False)
print("\nSales by Category:")
print(category_sales)

# 4. Payment Method Analysis
payment_methods = df['Type'].value_counts()
print("\nPayment Method Distribution:")
print(payment_methods)

# 5. Time-based Analysis
df['Hour'] = df['TransDate'].dt.hour
hourly_sales = df.groupby('Hour')['LineTotal'].sum()

# 6. Product Quantity Analysis
product_quantities = df.groupby('Product')['RQty'].sum().sort_values(ascending=False).head(15)
print("\nTop 15 Products by Quantity Sold:")
print(product_quantities)

# Create visualizations
plt.figure(figsize=(15, 10))

# 1. Location Sales
plt.subplot(2, 2, 1)
location_sales.plot(kind='bar')
plt.title('Sales by Location')
plt.xticks(rotation=45)
plt.tight_layout()

# 2. Top Products
plt.subplot(2, 2, 2)
product_sales.plot(kind='bar')
plt.title('Top 10 Products by Revenue')
plt.xticks(rotation=45)
plt.tight_layout()

# 3. Category Sales
plt.subplot(2, 2, 3)
category_sales.plot(kind='pie', autopct='%1.1f%%')
plt.title('Sales by Category')

# 4. Hourly Sales Pattern
plt.subplot(2, 2, 4)
hourly_sales.plot(kind='line', marker='o')
plt.title('Hourly Sales Pattern')
plt.xlabel('Hour of Day')
plt.ylabel('Total Sales')

plt.tight_layout()
plt.savefig('vending_machine_analysis.png')
plt.close()

# Create a separate figure for product quantities
plt.figure(figsize=(15, 8))
product_quantities.plot(kind='bar')
plt.title('Top 15 Products by Quantity Sold')
plt.xlabel('Product')
plt.ylabel('Quantity Sold')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('product_quantities.png')
plt.close()

# Additional Analysis
print("\nAverage Transaction Value:", df['LineTotal'].mean())
print("Total Number of Transactions:", len(df))
print("Total Revenue:", df['LineTotal'].sum())

# Save summary statistics to a text file
with open('vending_machine_summary.txt', 'w') as f:
    f.write("Vending Machine Sales Analysis Summary\n")
    f.write("=====================================\n\n")
    f.write(f"Total Revenue: ${df['LineTotal'].sum():,.2f}\n")
    f.write(f"Total Transactions: {len(df):,}\n")
    f.write(f"Average Transaction Value: ${df['LineTotal'].mean():,.2f}\n\n")
    
    f.write("Top 5 Locations by Revenue:\n")
    f.write(location_sales.head().to_string())
    f.write("\n\n")
    
    f.write("Top 5 Products by Revenue:\n")
    f.write(product_sales.head().to_string())
    f.write("\n\n")
    
    f.write("Top 15 Products by Quantity Sold:\n")
    f.write(product_quantities.to_string())
    f.write("\n\n")
    
    f.write("Sales by Category:\n")
    f.write(category_sales.to_string())
    f.write("\n\n")
    
    f.write("Payment Method Distribution:\n")
    f.write(payment_methods.to_string()) 
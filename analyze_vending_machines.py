import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np

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
payment_stats = df.groupby('Type').agg({
    'LineTotal': ['sum', 'mean', 'count']
}).reset_index()
payment_stats.columns = ['Payment_Type', 'Total_Sales', 'Average_Transaction', 'Transaction_Count']

# 6. Product Quantity Analysis
product_quantities = df.groupby('Product')['RQty'].sum().sort_values(ascending=False).head(15)
print("\nTop 15 Products by Quantity Sold:")
print(product_quantities)

# Create visualizations
plt.figure(figsize=(15, 8))

# 1. Top 10 Products by Quantity
plt.subplot(1, 2, 1)
product_quantities = df.groupby('Product')['RQty'].sum().sort_values(ascending=False).head(10)
product_quantities.plot(kind='bar', color='#2ecc71')
plt.title('Top 10 Products by Quantity Sold', pad=20, fontsize=12)
plt.xlabel('Product', labelpad=10)
plt.ylabel('Quantity Sold', labelpad=10)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 2. Sales by Category
plt.subplot(1, 2, 2)
category_sales = df.groupby('Category')['LineTotal'].sum().sort_values(ascending=False)
colors = ['#2ecc71', '#3498db', '#e74c3c', '#f1c40f']
plt.pie(category_sales, 
        labels=category_sales.index,
        autopct='%1.1f%%',
        colors=colors,
        explode=[0.05] + [0] * (len(category_sales)-1))
plt.title('Sales Distribution by Category', pad=20, fontsize=12)

# Add total sales amount to the pie chart
total_sales = category_sales.sum()
plt.text(0, -1.2, f'Total Sales: ${total_sales:,.2f}', 
         ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('vending_machine_analysis.png', bbox_inches='tight', dpi=300)
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
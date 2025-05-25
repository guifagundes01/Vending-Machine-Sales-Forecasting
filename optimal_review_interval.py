import pandas as pd
import numpy as np
from datetime import datetime

def calculate_optimal_review_interval(df, annual_holding_rate=0.2, trip_cost=50):
    """
    Calculate the optimal review interval for a group of items.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the sales data
    annual_holding_rate : float
        Annual holding cost rate (as a decimal)
    trip_cost : float
        Cost of a single replenishment trip (S0)
    
    Returns:
    --------
    dict
        Dictionary containing the optimal review interval and related metrics
    """
    # Calculate daily demand for each product
    # daily_demand = df.groupby('Product')['RQty'].sum() / df['TransDate'].nunique()

    
    annual_demand = df.groupby('Product')['RQty'].sum()
        
    # Calculate unit cost (assuming 40% margin)
    # This is an assumption - in reality, you would need actual cost data
    unit_costs = df.groupby('Product').apply(
        lambda x: (x['LineTotal'].sum() / x['RQty'].sum()) * 0.6
    )
    
    # Calculate annual holding cost per unit
    annual_holding_cost = unit_costs * annual_holding_rate
    
    # Calculate Di * Hi for each product
    demand_holding_cost = annual_demand * annual_holding_cost
    
    # Calculate the optimal review interval
    sum_demand_holding = demand_holding_cost.sum()
    optimal_interval = np.sqrt((2 * trip_cost) / sum_demand_holding)
    
    # Convert to days
    optimal_interval_days = optimal_interval * 365
    
    # Calculate additional metrics
    total_annual_demand = annual_demand.sum()
    total_annual_holding_cost = sum_demand_holding
    total_annual_ordering_cost = (trip_cost * 365) / optimal_interval_days
    
    return {
        'optimal_interval_years': optimal_interval,
        'optimal_interval_days': optimal_interval_days,
        'total_annual_demand': total_annual_demand,
        'total_annual_holding_cost': total_annual_holding_cost,
        'total_annual_ordering_cost': total_annual_ordering_cost,
        'product_details': pd.DataFrame({
            'Annual_Demand': annual_demand,
            'Unit_Cost': unit_costs,
            'Annual_Holding_Cost': annual_holding_cost,
            'Demand_Holding_Cost': demand_holding_cost
        })
    }

def main():
    # Read the data
    df = pd.read_csv('vending_machine_sales.csv')
    df['TransDate'] = pd.to_datetime(df['TransDate'])
    
    # Calculate optimal review interval
    results = calculate_optimal_review_interval(df)
    
    # Print results
    print("\nOptimal Review Interval Analysis")
    print("===============================")
    print(f"Optimal Review Interval: {results['optimal_interval_days']:.1f} days")
    print(f"Total Annual Demand: {results['total_annual_demand']:,.0f} units")
    print(f"Total Annual Holding Cost: ${results['total_annual_holding_cost']:,.2f}")
    print(f"Total Annual Ordering Cost: ${results['total_annual_ordering_cost']:,.2f}")
    
    # Save detailed results to CSV
    results['product_details'].to_csv('optimal_review_interval_details.csv')
    print("\nDetailed product-level analysis saved to 'optimal_review_interval_details.csv'")

if __name__ == "__main__":
    main() 
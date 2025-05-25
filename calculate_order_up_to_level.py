import pandas as pd
import numpy as np
from scipy import stats

def calculate_order_up_to_level(forecast_data, service_level=0.95):
    """
    Calculate the Order-Up-To Level (S) for each product considering a 2-month cycle.
    
    Parameters:
    -----------
    forecast_data : pandas.DataFrame
        DataFrame containing forecast data with columns:
        - Product: Product identifier
        - ARIMA_M1: Forecasted demand for first month
        - ARIMA_M2: Forecasted demand for second month
        - ARIMA_MSE: Mean Squared Error of the forecast
    service_level : float
        Desired service level (default 0.95 for 95% service level)
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing Order-Up-To levels and related metrics
    """
    # Create a copy of the input DataFrame to avoid SettingWithCopyWarning
    result_df = forecast_data.copy()
    
    # Calculate Z-score for the given service level
    z_score = stats.norm.ppf(service_level)
    
    # Calculate RMSE from MSE
    result_df.loc[:, 'RMSE'] = np.sqrt(result_df['ARIMA_MSE'])
    
    # Calculate total demand for 2-month period
    result_df.loc[:, 'Total_Demand_2Months'] = result_df['ARIMA_M1'] + result_df['ARIMA_M2']
    
    # Calculate Safety Stock for 2-month period
    # Using the formula: SS = Z * RMSE * sqrt(2)
    result_df.loc[:, 'Safety_Stock'] = z_score * result_df['RMSE'] * np.sqrt(2)
    
    # Calculate Order-Up-To Level (S)
    result_df.loc[:, 'Order_Up_To_Level'] = result_df['Total_Demand_2Months'] + result_df['Safety_Stock']
    
    # Round the values to whole numbers since we can't order fractional units
    result_df.loc[:, 'Order_Up_To_Level'] = np.ceil(result_df['Order_Up_To_Level'])
    result_df.loc[:, 'Safety_Stock'] = np.ceil(result_df['Safety_Stock'])
    result_df.loc[:, 'Total_Demand_2Months'] = np.ceil(result_df['Total_Demand_2Months'])
    
    return result_df

def main():
    try:
        # Read the data
        sales_df = pd.read_csv('vending_machine_sales.csv')
        forecast_df = pd.read_csv('output/all_methods_forecast_summary.csv')
        
        target_products = [
            'SunChips Multigrain - Harvest Cheddar',
            'CheezIt - Original',
            'SunChips Multigrain - Salsa'
        ]
        
        # Filter forecast data for target products
        product_forecasts = forecast_df[forecast_df['Product'].isin(target_products)].copy()
        
        # Calculate Order-Up-To levels
        results_95 = calculate_order_up_to_level(product_forecasts, service_level=0.95)
        results_99 = calculate_order_up_to_level(product_forecasts, service_level=0.99)
        
        # Combine results
        final_results = pd.DataFrame({
            'Product': results_95['Product'],
            'Forecast_Month1': results_95['ARIMA_M1'],
            'Forecast_Month2': results_95['ARIMA_M2'],
            'Total_Demand_2Months': results_95['Total_Demand_2Months'],
            'RMSE': results_95['RMSE'],
            'Safety_Stock_95': results_95['Safety_Stock'],
            'Order_Up_To_Level_95': results_95['Order_Up_To_Level'],
            'Safety_Stock_99': results_99['Safety_Stock'],
            'Order_Up_To_Level_99': results_99['Order_Up_To_Level']
        })
        
        # Sort by Order-Up-To Level
        final_results = final_results.sort_values('Order_Up_To_Level_95', ascending=False)
        
        # Create formatted text output
        with open('output/inventory_management/order_up_to_levels_snacks.txt', 'w') as f:
            f.write("Order-Up-To Level Analysis for Top Snacks (2-Month Cycle)\n")
            f.write("===================================================\n\n")
            f.write(f"Total Products Analyzed: {len(final_results)}\n\n")
            
            f.write("Detailed Product Analysis (95% Service Level):\n")
            f.write("-" * 100 + "\n")
            f.write(f"{'Product':<40} {'Month1':>8} {'Month2':>8} {'Total Demand':>12} {'Safety Stock':>12} {'Order-Up-To':>12}\n")
            f.write("-" * 100 + "\n")
            
            for _, row in final_results.iterrows():
                f.write(f"{row['Product']:<40} {row['Forecast_Month1']:>8.1f} {row['Forecast_Month2']:>8.1f} "
                       f"{row['Total_Demand_2Months']:>12.1f} {row['Safety_Stock_95']:>12.1f} {row['Order_Up_To_Level_95']:>12.1f}\n")
            
            f.write("\nDetailed Product Analysis (99% Service Level):\n")
            f.write("-" * 100 + "\n")
            f.write(f"{'Product':<40} {'Month1':>8} {'Month2':>8} {'Total Demand':>12} {'Safety Stock':>12} {'Order-Up-To':>12}\n")
            f.write("-" * 100 + "\n")
            
            for _, row in final_results.iterrows():
                f.write(f"{row['Product']:<40} {row['Forecast_Month1']:>8.1f} {row['Forecast_Month2']:>8.1f} "
                       f"{row['Total_Demand_2Months']:>12.1f} {row['Safety_Stock_99']:>12.1f} {row['Order_Up_To_Level_99']:>12.1f}\n")
        
        # Print summary
        print("\nOrder-Up-To Level Analysis for Top Snacks (2-Month Cycle)")
        print("===================================================")
        print(f"Total Products Analyzed: {len(final_results)}")
        print("\nResults have been saved to 'order_up_to_levels_snacks.txt'")
        
    except FileNotFoundError as e:
        print(f"Error: Could not find required file: {str(e)}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 
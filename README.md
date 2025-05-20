# Vending Machine Sales Forecasting

This project implements various time series forecasting methods to predict sales for vending machines. The analysis includes multiple forecasting techniques such as Moving Average, Exponential Smoothing, and ARIMA models.

## Project Structure

- `vending_machine_sales.csv`: Main dataset containing vending machine sales data
- `moving_average.py`: Implementation of Moving Average forecasting
- `exponential_smoothing.py`: Implementation of Exponential Smoothing forecasting
- `arima_forecasting.py`: Implementation of ARIMA forecasting
- `all_methods_forecast_summary.py`: Comparison of different forecasting methods
- `forecasting_model_analysis.py`: Analysis of forecasting model performance
- `top_products_arima.py`: ARIMA forecasting for top products
- `top10_products_exponential_smoothing.py`: Exponential smoothing for top products
- `top10_products_ma.py`: Moving average for top products
- `analyze_vending_machines.py`: Analysis of vending machine data

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

The project provides several scripts for different forecasting methods:

1. Moving Average Forecasting:
```bash
python moving_average.py
```

2. Exponential Smoothing:
```bash
python exponential_smoothing.py
```

3. ARIMA Forecasting:
```bash
python arima_forecasting.py
```

4. Compare all methods:
```bash
python all_methods_forecast_summary.py
```

## Output

The forecasting results and visualizations are saved in the `output/` directory.

## Requirements

- Python 3.8+
- pandas >= 2.0.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- numpy >= 1.24.0
- statsmodels >= 0.14.0
- scikit-learn >= 1.3.0
- pmdarima >= 2.0.0

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
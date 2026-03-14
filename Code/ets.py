import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.api import ExponentialSmoothing
import utils
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

FORECAST_STEPS = 84 # forecast till 2024
TEST_DURATION = 24  # 2 years for test set

def test_ets(data_series, test_duration=TEST_DURATION):
    print("\nPART 1: ETS Test Set Error Calculation\n")
    
    # Split data
    train_data, test_data = utils.create_split(data_series)
    print(f"Training on {len(train_data)} data points, testing on {len(test_data)}.\n\n")

    print("\n Training ETS Model (Holt-Winters) on TRAIN data ")
    ets_model_test = ExponentialSmoothing(
        train_data, 
        seasonal_periods=12, 
        trend='add', 
        seasonal='mul', 
        initialization_method='estimated'
    ).fit()
    print("ETS test model fitting complete.\n\n")

    forecast = ets_model_test.forecast(steps=test_duration)
    pred_mean = forecast
    
    utils.print_metrics(test_data.values, pred_mean.values, "ETS")

def ets_pred(data_series):
    print("\n\nExponential Smoothing Forecast till 2024\n")
    
    print("\n Training ETS Model (Holt-Winters) on FULL data ")
    ets_model = ExponentialSmoothing(
        data_series, 
        seasonal_periods=12, 
        trend='add', 
        seasonal='mul', 
        initialization_method='estimated'
    ).fit()
    
    print("ETS model fitting complete.\n")
    print(ets_model.summary())

    # Define forecast period
    last_date = data_series.index[-1]
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=FORECAST_STEPS, freq='MS') 
    
    # Generate forecast
    forecast = ets_model.forecast(steps=FORECAST_STEPS)
    pred_mean = forecast
    
    forecast_series = pd.Series(pred_mean.values, index=future_dates, name='ETS_Forecast')
    
    # Combine actuals and forecast for plotting and saving (Horizontal concat for robustness)
    df_full = data_series.rename('Actual_Value').to_frame()
    df_pred = pd.DataFrame({'ETS_Forecast': pred_mean,}, index=future_dates)
    df_full = pd.concat([df_full, df_pred], axis=1)

    # Save to CSV
    df_full.to_csv('ets_full_forecast_and_actuals.csv')
    print("\nSuccessfully saved combined ETS data to ets_full_forecast_and_actuals.csv")
    
    plt.figure(figsize=(15, 8))
    plt.plot(df_full.index, df_full['Actual_Value'], label='Actual Consumption', color='blue')
    plt.plot(future_dates, df_full.loc[future_dates, 'ETS_Forecast'], label='ETS Forecast', color='red', linestyle='--')
    plt.title('Electricity Consumption: Actuals vs. ETS Forecast', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axvline(x=future_dates[0], color='green', linestyle=':', linewidth=2, label='Forecast Start')
    plt.legend()
    plt.tight_layout()
    plot_filename = 'ets_full_forecast_plot.png'
    plt.savefig(plot_filename)
    plt.close()
    print(f"Successfully saved plot to {plot_filename}")

if __name__ == '__main__':
    data_series = utils.load_data()
    test_ets(data_series)
    ets_pred(data_series)
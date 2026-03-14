import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
import utils
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

FORECAST_YEARS = 7 
TEST_DURATION = 24 # 2 years for test set
FORECAST_STEPS = FORECAST_YEARS * 12

def plot_pred(df_full, df_pred, plot_filename='prophet_full_forecast_plot.png'):
    future_dates = df_pred.index
    
    plt.figure(figsize=(15, 8))
    plt.plot(df_full.index, df_full['Actual_Value'], label='Actual Consumption', color='blue')
    plt.plot(future_dates, df_full.loc[future_dates, 'Prophet_Forecast'], label='Prophet Forecast', color='red', linestyle='--') 
    plt.fill_between(future_dates, df_full.loc[future_dates, 'Prophet_Lower'], df_full.loc[future_dates, 'Prophet_Upper'], color='red', alpha=0.1, label='95% Confidence Interval')
    plt.title('Electricity Consumption: Actuals vs. Prophet Forecast', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axvline(x=future_dates[0], color='green', linestyle=':', linewidth=2, label='Forecast Start')
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_filename)
    plt.close()
    print(f"Successfully saved forecast plot to {plot_filename}")

#  Test Set Error Calculation 

def test_prophet(data_series, test_duration=TEST_DURATION):
    print("\nProphet Test Set Error Calculation\n")
    
    # Split data
    train, test = utils.create_split(data_series)

    print(f"Training on {len(train)} data points, testing on {len(test)}.")

    df_prophet_train = pd.DataFrame({'ds': train.index, 'y': train.values})

    model_test = Prophet(
        yearly_seasonality=True, 
        weekly_seasonality=False, 
        daily_seasonality=False
    )
    
    print("\n Training Prophet model on TRAIN data ")
    model_test.fit(df_prophet_train)

    # Create future dataframe for the test period
    future_test = model_test.make_future_dataframe(periods=test_duration, freq='MS', include_history=False)

    # Generate forecast
    test_pred_df = model_test.predict(future_test)
    test_pred_values = test_pred_df['yhat'].values
    test_values = test.values
    
    # Calculate and print metrics
    utils.print_metrics(test_values, test_pred_values, "Prophet")

#  Full Data Forecast 

def full_forecast(data_series):
    print("\nFull forecast using Prophet till 2024")
    print(f"Training on {len(data_series)} total observations for final forecast.")

    df_prophet = pd.DataFrame({
        'ds': data_series.index,  
        'y': data_series.values
    })

    # Fit Prophet Model
    model = Prophet(
        yearly_seasonality=True, 
        weekly_seasonality=False, 
        daily_seasonality=False
    )
    
    print("\n Training Prophet model on FULL data ")
    model.fit(df_prophet)

    # Create Future DataFrame
    future = model.make_future_dataframe(periods=FORECAST_STEPS, freq='MS', include_history=False)

    # Generate Forecast
    final_forecast = model.predict(future)
    
    # Combine and Save Data
    df_pred = final_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(
        columns={'ds': 'DATE', 'yhat': 'Prophet_Forecast', 
                 'yhat_lower': 'Prophet_Lower', 'yhat_upper': 'Prophet_Upper'}
    ).set_index('DATE')

    df_full = data_series.rename('Actual_Value').to_frame()
    df_full = pd.concat([df_full, df_pred], axis=1)
    
    csv_path = 'prophet_forecast.csv'
    df_full.to_csv(csv_path)
    print(f"\nSuccessfully saved combined Prophet data to {csv_path}")

    plot_pred(df_full, df_pred)

    fig_comp_final = model.plot_components(final_forecast)
    fig_comp_final.savefig('prophet_final_components_plot.png')
    plt.close(fig_comp_final)
    print("Successfully saved Prophet components plot to prophet_final_components_plot.png")


if __name__ == '__main__':
    data_series = utils.load_data()
    test_prophet(data_series, test_duration=TEST_DURATION)
    full_forecast(data_series)
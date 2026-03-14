import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pmdarima as pm
import statsmodels.api as sm
import warnings
import utils
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=ConvergenceWarning)

TEST_DURATION = 24  # 2 years for test set
FORECAST_STEPS = 84 # forecast till 2024

def best_sarima(train_data): #uses auto_arima to find the best SARIMA parameters on the training subset.
    print(f"Training data points for selection: {len(train_data)}\n")
    print("\nRunning auto_arima to find the best SARIMA model on training data...")

    model = pm.auto_arima(train_data, 
                               start_p=1, start_q=1,
                               test='adf',       
                               max_p=5, max_q=5,
                               m=12,             # 12-month seasonality
                               d=1,           
                               seasonal=True,    
                               start_P=0, 
                               D=1,           
                               trace=True,      
                               error_action='ignore',  
                               suppress_warnings=True, 
                               stepwise=True)
    
    print("\n Best Model Found on Train Split ")
    print(model.summary())
    return model

def test_sarima(data_series, test_duration=TEST_DURATION): # Fits SARIMA on training set and evaluates on test set.
    print("SARIMA Test Error Calculation")
    
    # split data
    train_data, test_data = utils.create_split(data_series)
    print(f"Training on {len(train_data)} data points, testing on {len(test_data)}.")

    # find best model on training data
    best_model = best_sarima(train_data)
    best_order = best_model.order
    best_seasonal_order = best_model.seasonal_order
    
    print(f"\nRefitting best model {best_order}{best_seasonal_order} on training data...")
    
    # 3. Fit the best model found on the training data
    try:
        model_test = sm.tsa.SARIMAX(train_data,
                                           order=best_order,
                                           seasonal_order=best_seasonal_order,
                                           enforce_stationarity=False,
                                           enforce_invertibility=False).fit(disp=False)
    except Exception as e:
        print(f"Error fitting test SARIMA model: {e}. Defaulting to (2,1,3)(2,1,1)[12]")
        best_order = (2, 1, 3)
        best_seasonal_order = (2, 1, 1, 12)
        model_test = sm.tsa.SARIMAX(train_data,
                                           order=best_order,
                                           seasonal_order=best_seasonal_order,
                                           enforce_stationarity=False,
                                           enforce_invertibility=False).fit(disp=False)

    forecast = model_test.get_forecast(steps=test_duration)
    pred_mean = forecast.predicted_mean
    
    # calculate and print metrics
    utils.print_metrics(test_data.values, pred_mean.values, "SARIMA")

def sarima_pred(data_series, n_future_steps=FORECAST_STEPS):
    print("Full Forecast till 2024 Using SARIMA")
    
    print("\n Training SARIMA(2,1,3)(2,1,1)[12] on FULL data ")
    model_final = sm.tsa.SARIMAX(data_series,
                                       order=(2, 1, 3),
                                       seasonal_order=(2, 1, 1, 12),
                                       enforce_stationarity=False,
                                       enforce_invertibility=False).fit(disp=False)
    print("SARIMA model fitting complete.")

    last_date = data_series.index[-1]
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), 
                                 periods=n_future_steps, 
                                 freq='MS') 
    
    # Generate forecast
    forecast = model_final.get_forecast(steps=n_future_steps)
    pred_mean = forecast.predicted_mean
    forecast_ci = forecast.conf_int()
    
    # Combine forecast series and confidence intervals into a forecast DataFrame
    df_pred = pd.DataFrame({
        'SARIMA_Forecast': pred_mean,
        'Forecast_Upper': forecast_ci.iloc[:, 1],
        'Forecast_Lower': forecast_ci.iloc[:, 0]
    }, index=future_dates)
    
    # Combine Actuals (Series) and Forecast (DataFrame) horizontally (axis=1)
    df_full = pd.concat([
        data_series.rename('Actual_Value'), 
        df_pred                          
    ], axis=1)

    df_full.to_csv('sarima_full_forecast_and_actuals.csv')
    print("\nSuccessfully saved combined SARIMA data to sarima_full_forecast_and_actuals.csv")
    
    # Plotting
    plt.figure(figsize=(15, 8))
    plt.plot(df_full.index, df_full['Actual_Value'], label='Actual Consumption', color='blue')
    plt.plot(df_full.index, df_full['SARIMA_Forecast'], label='SARIMA Forecast', color='red', linestyle='--') 
    plt.fill_between(future_dates, 
                     df_full.loc[future_dates, 'Forecast_Lower'], 
                     df_full.loc[future_dates, 'Forecast_Upper'], 
                     color='red', alpha=0.1, label='95% Confidence Interval')
    
    plt.title('Electricity Consumption: Actuals vs. SARIMA Forecast', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axvline(x=future_dates[0], color='green', linestyle=':', linewidth=2, label='Forecast Start')
    plt.legend()
    plt.tight_layout()
    plot_filename = 'sarima_full_forecast_plot.png'
    plt.savefig(plot_filename)
    plt.close()
    print(f"Successfully saved plot to {plot_filename}")

    return model_final, df_full

if __name__ == '__main__':
    data_series = utils.load_data()
    test_sarima(data_series)
    sarima_pred(data_series)
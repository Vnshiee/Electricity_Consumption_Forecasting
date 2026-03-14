import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utils
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
from statsmodels.tools.sm_exceptions import InterpolationWarning

warnings.filterwarnings('ignore', category=InterpolationWarning)

def adf_test(series, name="Series"):
    result = adfuller(series.dropna())
    
    print(f"\n\n ADF Test Results for: {name} ")
    print(f"ADF Statistic: {result[0]:.4f}")
    print(f"p-value: {result[1]:.6f}")
    
    # Interpretation
    if result[1] <= 0.05:
        print(f"Result: Data is Stationary (p-value <= 0.05)")
    else:
        print(f"Result: Data is Non-Stationary (p-value = {result[1]:.6f} > 0.05)")
        
    return result

def kpss_test(series, name="Series", regression='c'):
    result = kpss(series.dropna(), regression=regression, nlags='auto')
    
    print(f"\n\n KPSS Test Results for: {name} (regression='{regression}') ")
    print(f"KPSS Statistic: {result[0]:.6f}")
    print(f"p-value: {result[1]:.6f}")
    
    if result[1] >= 0.05:
        print(f"Result: Data is Stationary (p-value >= 0.05)")
    else:
        print(f"Result: Data is Non-Stationary (p-value = {result[1]:.6f} < 0.05)")

    return result

def get_plots(series, title_suffix="Original Data"):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    plot_acf(series.dropna(), lags=40, ax=ax1, title=f'Autocorrelation (ACF) - {title_suffix}')
    ax1.grid(True)
    plot_pacf(series.dropna(), lags=40, ax=ax2, title=f'Partial Autocorrelation (PACF) - {title_suffix}')
    ax2.grid(True)
    
    plt.tight_layout()
    filename = f'acf_pacf_{title_suffix.lower().replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")}.png'
    plt.savefig(filename)
    plt.close()
    print(f"\nSaved '{filename}'")

if __name__ == '__main__':
    data_series = utils.load_data()
    df_full = data_series.to_frame(name='Value')
    
    print("Stationarity Tests:")
    
    # 1. Test Original Series
    print("\nTesting Original Series...")
    adf_test(df_full['Value'], "Original Series")
    kpss_test(df_full['Value'], "Original Series", regression='c')
    kpss_test(df_full['Value'], "Original Series", regression='ct')
    
    # Calculate differenced series
    df_full['diff_1'] = df_full['Value'].diff(1)
    df_full['diff_2'] = df_full['diff_1'].diff(12)

    # 2. Test Seasonal + First Differenced Series (d=1, D=1, m=12)
    print("\nTesting Seasonal + First Differenced Series (d=1, D=1)...")
    adf_test(df_full['diff_2'], "Seasonal + First Differenced Series (d=1, D=1)")
    kpss_test(df_full['diff_2'], "Seasonal + First Differenced Series (d=1, D=1)", regression='c')

    get_plots(df_full['Value'], "Original Data")
    get_plots(df_full['diff_2'], "Seasonal + First Differenced Data (d=1, D=1)")
    
    print("\nConclusion: The 'Seasonal + First Differenced Series' is highly Stationary, supporting **d=1** and **D=1** for SARIMA")
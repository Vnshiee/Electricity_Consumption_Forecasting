import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utils
from statsmodels.tsa.seasonal import seasonal_decompose
from datetime import datetime

def get_month_name(month_number):
    return datetime(2000, month_number, 1).strftime('%b')

def visualize(data_series):
    df = data_series.to_frame(name='Value').reset_index().rename(columns={'index': 'DATE'})
    df['Year'] = df['DATE'].dt.year
    df['Month'] = df['DATE'].dt.month

    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    # Main Time Series Plot with Trend Line
    axes[0].plot(df['DATE'], df['Value'], linewidth=1.5, color='steelblue')
    z = np.polyfit(range(len(df)), df['Value'], 1)
    p = np.poly1d(z)
    axes[0].plot(df['DATE'], p(range(len(df))), "r--", linewidth=2, label=f'Trend Line (slope={z[0]:.3f})')
    axes[0].set_title('Electricity Consumption Over Time', fontweight='bold')
    axes[0].legend()
    
    # Box Plot for Monthly Variation (using the Series directly)
    data_by_month = [data_series[data_series.index.month == m] for m in range(1, 13)]
    ax = axes[1].boxplot(data_by_month, patch_artist=True, medianprops={'color': 'red'})
    month_names = [get_month_name(m) for m in range(1, 13)]
    axes[1].set_xticklabels(month_names)
    axes[1].set_title('Consumption Distribution by Month', fontweight='bold')

    # Year-over-Year Plot (Seasonal)
    monthly_data = df.pivot_table(index='Month', columns='Year', values='Value', aggfunc='mean')
    monthly_data.plot(ax=axes[2], colormap='viridis', legend=False) 
    axes[2].set_xticks(range(1, 13))
    axes[2].set_xticklabels(month_names)
    axes[2].set_title('Seasonal Pattern: Consumption by Month Across Years', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('time_series_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()


def decomp(data_series):
    print("\nTime Series Decomposition:")
    
    df_full = data_series.to_frame(name='Value')
    decomposition_add = seasonal_decompose(df_full['Value'], model='additive', period=12, extrapolate_trend='freq')
        
    # Calculate Trend and Seasonality Strength
    var_residual_add = np.var(decomposition_add.resid.dropna())
    var_detrended_add = np.var((decomposition_add.resid + decomposition_add.seasonal).dropna())
    strength_trend_add = max(0, 1 - (var_residual_add / np.var((decomposition_add.trend + decomposition_add.resid).dropna())))
    strength_seasonality_add = max(0, 1 - (var_residual_add / var_detrended_add))

    print("\nAdditive Model:")
    print(f" Trend Strength: {strength_trend_add:.4f}")
    print(f" Seasonality Strength: {strength_seasonality_add:.4f}")

    decomposition_mult = seasonal_decompose(df_full['Value'], model='multiplicative', period=12, extrapolate_trend='freq')
    print("\nMultiplicative Model:")
    print(f" Seasonal Multiplier Range: {decomposition_mult.seasonal.min():.4f} to {decomposition_mult.seasonal.max():.4f}")
    
    if 'decomposition_add' in locals():
        fig = decomposition_add.plot()
        fig.set_size_inches(15, 12)
        plt.savefig('time_series_decomposition.png')
        plt.close()
        print("\nDecomposition plot saved as: time_series_decomposition.png")

if __name__ == '__main__':
    data_series = utils.load_data()
    visualize(data_series)
    decomp(data_series)
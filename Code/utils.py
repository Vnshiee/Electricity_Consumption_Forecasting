import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

PATH = '27.csv'
TEST_DURATION = 24

def load_data(path=r'D:\Projects\DSP_project_and_Ass\Final_Submission\Final_Submission\Code\27.csv'):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df['DATE'] = pd.to_datetime(df['DATE'], format='%m-%d-%Y')
    df = df.set_index('DATE')
    data_series = df['Value'].copy()
    date_range = pd.date_range(start=data_series.index.min(), end=data_series.index.max(), freq='MS')
    data_series = data_series.reindex(date_range)
    
    return data_series

def create_split(data_series, test_duration=TEST_DURATION):
    train = data_series.iloc[:-test_duration]
    test = data_series.iloc[-test_duration:]
    return train, test

def print_metrics(y_true, y_pred, name="Model"):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = 100*mean_absolute_percentage_error(y_true, y_pred)
    
    print(f"\n Test Set Metrics for {name} ({TEST_DURATION} Months) ")
    print(f"    MAE:  {mae:.4f}")
    print(f"    RMSE: {rmse:.4f}")
    print(f"    MAPE: {mape:.4f} %")
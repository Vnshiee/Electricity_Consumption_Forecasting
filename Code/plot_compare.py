import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

# Model Imports
from statsmodels.tsa.api import ExponentialSmoothing
from prophet import Prophet
import pmdarima as pm
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ==========================================
# 1. UTILS & DATA LOADING (From utils.py)
# ==========================================
TEST_DURATION = 24

def load_and_split_data(path=r'D:\Projects\DSP_project_and_Ass\Final_Submission\Final_Submission\Code\27.csv'):
    # Load
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df['DATE'] = pd.to_datetime(df['DATE'], format='%m-%d-%Y')
    df = df.set_index('DATE')
    data_series = df['Value'].copy()
    
    # Reindex to MS freq
    date_range = pd.date_range(start=data_series.index.min(), 
                               end=data_series.index.max(), 
                               freq='MS')
    data_series = data_series.reindex(date_range)
    
    # Split
    train = data_series.iloc[:-TEST_DURATION]
    test = data_series.iloc[-TEST_DURATION:]
    return train, test

# ==========================================
# 2. PLOTTING HELPER
# ==========================================
def plot_comparison(train, test, forecast, model_name):
    plt.figure(figsize=(14, 6))
    
    # Plot last 5 years of training to keep it readable, or full if short
    train_subset = train.iloc[-60:] 
    
    plt.plot(train_subset.index, train_subset.values, label='Training Data (Last 5 Yrs)', color='gray', alpha=0.7)
    plt.plot(test.index, test.values, label='Test Data (Actuals)', color='blue', linewidth=2)
    plt.plot(test.index, forecast, label=f'{model_name} Forecast', color='red', linestyle='--', linewidth=2)
    
    plt.title(f'{model_name}: Training vs Test vs Forecast', fontsize=14)
    plt.axvline(x=test.index[0], color='green', linestyle=':', label='Train/Test Split')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{model_name}_forecast_comparison.png')
    plt.show()

# ==========================================
# 3. MODEL: ETS (Exponential Smoothing)
# ==========================================
def run_ets(train, test):
    print("Running ETS...")
    # Logic from ets.py: trend='add', seasonal='mul'
    model = ExponentialSmoothing(
        train, 
        seasonal_periods=12, 
        trend='add', 
        seasonal='mul', 
        initialization_method='estimated'
    ).fit()
    
    forecast = model.forecast(steps=len(test))
    plot_comparison(train, test, forecast, "ETS")

# ==========================================
# 4. MODEL: SARIMA
# ==========================================
def run_sarima(train, test):
    print("Running SARIMA (this may take a moment)...")
    # Logic from sarima.py: auto_arima
    model = pm.auto_arima(train, 
                          start_p=1, start_q=1,
                          max_p=5, max_q=5,
                          m=12, d=1, D=1,
                          seasonal=True,
                          trace=False,
                          error_action='ignore',  
                          suppress_warnings=True)
    
    forecast = model.predict(n_periods=len(test))
    plot_comparison(train, test, forecast, "SARIMA")

# ==========================================
# 5. MODEL: PROPHET
# ==========================================
def run_prophet(train, test):
    print("Running Prophet...")
    # Prepare data for Prophet (ds, y)
    df_train = train.reset_index()
    df_train.columns = ['ds', 'y']
    
    # Logic from FBprophet.py
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    m.fit(df_train)
    
    # Create future frame for test duration
    future = m.make_future_dataframe(periods=len(test), freq='MS')
    forecast_full = m.predict(future)
    
    # Extract only the test period predictions
    forecast_values = forecast_full.iloc[-len(test):]['yhat'].values
    
    plot_comparison(train, test, forecast_values, "Prophet")

# ==========================================
# 6. MODEL: LSTM
# ==========================================
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.linear(lstm_out[:, -1, :])

def run_lstm(train, test):
    print("Running LSTM...")
    
    # Config from lstm.py
    look_back = 12
    hidden_size = 50
    epochs = 100  # Simplified for quick plotting
    
    # Scaling
    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_scaled = scaler.fit_transform(train.values.reshape(-1, 1))
    
    # Create Sequences
    X_train, y_train = [], []
    for i in range(len(train_scaled) - look_back):
        X_train.append(train_scaled[i:i+look_back])
        y_train.append(train_scaled[i+look_back])
    
    X_train = torch.FloatTensor(np.array(X_train))
    y_train = torch.FloatTensor(np.array(y_train))
    
    # Train
    model = LSTMModel(hidden_size=hidden_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
    # Recursive Prediction for Test Set
    model.eval()
    test_predictions = []
    
    # Initialize with last sequence from training
    curr_seq = torch.FloatTensor(train_scaled[-look_back:]).view(1, look_back, 1)
    
    for _ in range(len(test)):
        with torch.no_grad():
            pred_val = model(curr_seq)
            test_predictions.append(pred_val.item())
            
            # Update sequence: remove first, append prediction
            pred_reshaped = pred_val.view(1, 1, 1)
            curr_seq = torch.cat((curr_seq[:, 1:, :], pred_reshaped), dim=1)
            
    # Inverse Transform
    forecast_values = scaler.inverse_transform(np.array(test_predictions).reshape(-1, 1)).flatten()
    
    plot_comparison(train, test, forecast_values, "LSTM")

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    try:
        train_data, test_data = load_and_split_data()
        
        run_ets(train_data, test_data)
        run_sarima(train_data, test_data)
        run_prophet(train_data, test_data)
        run_lstm(train_data, test_data)
        
    except FileNotFoundError:
        print("Error: '27.csv' not found. Please check the file path.")
    except Exception as e:
        print(f"An error occurred: {e}")
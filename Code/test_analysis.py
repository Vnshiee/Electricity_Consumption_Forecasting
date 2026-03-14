import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.api import ExponentialSmoothing
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# 1. Load and Split Data
def load_and_split_data(path=r'D:\Projects\DSP_project_and_Ass\Final_Submission\Final_Submission\Code\27.csv'):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df['DATE'] = pd.to_datetime(df['DATE'], format='%m-%d-%Y')
    df = df.set_index('DATE')
    data_series = df['Value'].copy()
    
    # Reindex to MS freq to ensure continuity
    date_range = pd.date_range(start=data_series.index.min(), 
                               end=data_series.index.max(), 
                               freq='MS')
    data_series = data_series.reindex(date_range)
    
    TEST_DURATION = 24
    train = data_series.iloc[:-TEST_DURATION]
    test = data_series.iloc[-TEST_DURATION:]
    return train, test

train, test = load_and_split_data()

# Metric Container
results = {}

def calculate_metrics(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    return {'MAE': mae, 'RMSE': rmse, 'MAPE (%)': mape}

# ==========================================
# Model 1: ETS
# ==========================================
try:
    ets_model = ExponentialSmoothing(
        train, 
        seasonal_periods=12, 
        trend='add', 
        seasonal='mul', 
        initialization_method='estimated'
    ).fit()
    ets_pred = ets_model.forecast(steps=len(test))
    results['ETS'] = calculate_metrics(test, ets_pred, 'ETS')
except Exception as e:
    results['ETS'] = {'Error': str(e)}

# ==========================================
# Model 2: SARIMA
# ==========================================
# Checking for pmdarima, else fallback to statsmodels
try:
    import pmdarima as pm
    sarima_model = pm.auto_arima(train, start_p=1, start_q=1,
                          max_p=3, max_q=3, m=12, d=1, D=1,
                          seasonal=True, trace=False,
                          error_action='ignore', suppress_warnings=True)
    sarima_pred = sarima_model.predict(n_periods=len(test))
    results['SARIMA'] = calculate_metrics(test, sarima_pred, 'SARIMA')
except ImportError:
    # Fallback to statsmodels if pmdarima is not installed
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    # Using a standard order often found by auto_arima for monthly data
    sarima_model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
                           enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    sarima_pred = sarima_model.forecast(steps=len(test))
    results['SARIMA'] = calculate_metrics(test, sarima_pred, 'SARIMA (Fixed Order)')
except Exception as e:
    results['SARIMA'] = {'Error': str(e)}

# ==========================================
# Model 3: Prophet
# ==========================================
try:
    from prophet import Prophet
    df_train = train.reset_index()
    df_train.columns = ['ds', 'y']
    
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    m.fit(df_train)
    
    future = m.make_future_dataframe(periods=len(test), freq='MS')
    forecast = m.predict(future)
    prophet_pred = forecast.iloc[-len(test):]['yhat'].values
    results['Prophet'] = calculate_metrics(test, prophet_pred, 'Prophet')
except ImportError:
    results['Prophet'] = {'Error': 'Prophet library not found'}
except Exception as e:
    results['Prophet'] = {'Error': str(e)}

# ==========================================
# Model 4: LSTM
# ==========================================
try:
    # Config
    look_back = 12
    hidden_size = 50
    epochs = 50 # Reduced for speed
    
    # Scale
    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_scaled = scaler.fit_transform(train.values.reshape(-1, 1))
    
    # Prepare Data
    X_train, y_train = [], []
    for i in range(len(train_scaled) - look_back):
        X_train.append(train_scaled[i:i+look_back])
        y_train.append(train_scaled[i+look_back])
    
    X_train = torch.FloatTensor(np.array(X_train))
    y_train = torch.FloatTensor(np.array(y_train))
    
    # Model
    class LSTMModel(nn.Module):
        def __init__(self):
            super(LSTMModel, self).__init__()
            self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
            self.linear = nn.Linear(hidden_size, 1)
        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            return self.linear(lstm_out[:, -1, :])
            
    model = LSTMModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Train
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        
    # Predict (Recursive)
    model.eval()
    test_preds = []
    curr_seq = torch.FloatTensor(train_scaled[-look_back:]).view(1, look_back, 1)
    
    for _ in range(len(test)):
        with torch.no_grad():
            pred_val = model(curr_seq)
            test_preds.append(pred_val.item())
            pred_reshaped = pred_val.view(1, 1, 1)
            curr_seq = torch.cat((curr_seq[:, 1:, :], pred_reshaped), dim=1)
            
    lstm_pred = scaler.inverse_transform(np.array(test_preds).reshape(-1, 1)).flatten()
    results['LSTM'] = calculate_metrics(test, lstm_pred, 'LSTM')

except Exception as e:
    results['LSTM'] = {'Error': str(e)}

# Display Results
results_df = pd.DataFrame(results).T
print(results_df)
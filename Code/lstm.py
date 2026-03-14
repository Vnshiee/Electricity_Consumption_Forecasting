import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import utils
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler

future_steps=84
look_back=12

print(" Starting LSTM Model Training")

data_series = utils.load_data()
train, test = utils.create_split(data_series)

val_size = int(0.1*len(train))

test_numpy = test.values
train_numpy = train.values[:-val_size]
val_numpy = train.values[-val_size:]

print(f"Data split:")
print(f"  Train: {len(train_numpy)} samples")
print(f"  Val:   {len(val_numpy)} samples")
print(f"  Test:  {len(test_numpy)} samples")

print("Scaling data...")
scaler = MinMaxScaler(feature_range=(-1, 1))
train_scaled = scaler.fit_transform(train_numpy.reshape(-1, 1))
val_scaled = scaler.transform(val_numpy.reshape(-1, 1))
test_scaled = scaler.transform(test_numpy.reshape(-1, 1))

#  Step 5: Create Sequences 
def create_sequences(data, look_back=12):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back)])
        y.append(data[i + look_back])
    return np.array(X), np.array(y)

print(f"Creating sequences with look_back={look_back}...")
X_train, y_train = create_sequences(train_scaled, look_back)
X_val, y_val = create_sequences(val_scaled, look_back)
X_test, y_test = create_sequences(test_scaled, look_back)

#  Step 6: Convert to Tensors 
X_train_tensor = torch.from_numpy(X_train).float()
y_train_tensor = torch.from_numpy(y_train).float()
X_val_tensor = torch.from_numpy(X_val).float()
y_val_tensor = torch.from_numpy(y_val).float()
X_test_tensor = torch.from_numpy(X_test).float()
y_test_tensor = torch.from_numpy(y_test).float()

print(f"Train tensor shape: {X_train_tensor.shape}")
print(f"Val tensor shape: {X_val_tensor.shape}")
print(f"Test tensor shape: {X_test_tensor.shape}")

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=50, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        h0 = torch.zeros(1, input_seq.size(0), self.hidden_layer_size)
        c0 = torch.zeros(1, input_seq.size(0), self.hidden_layer_size)
        lstm_out, _ = self.lstm(input_seq, (h0, c0))
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions

model = LSTMModel()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=120)   

print(model)

epochs = 2000
train_losses = []
val_losses = []

print(f"\n Starting Training for {epochs} epochs ")

for i in range(epochs):
    model.train() 
    optimizer.zero_grad()
    y_pred_train = model(X_train_tensor)
    loss_train = loss_function(y_pred_train, y_train_tensor)
    loss_train.backward()
    optimizer.step()
    train_losses.append(loss_train.item())
    
    #  Validation 
    model.eval() 
    with torch.no_grad():
        y_pred_val = model(X_val_tensor)
        loss_val = loss_function(y_pred_val, y_val_tensor)
        val_losses.append(loss_val.item())
    
    #  Step the scheduler 
    scheduler.step(loss_val)
    
    if (i+1) % 25 == 0: # Print less frequently
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {i+1}/{epochs} - Train Loss: {loss_train.item():.4f}, Val Loss: {loss_val.item():.4f}, LR: {current_lr:g}')

print(" Training Complete ")

# Define a file path for the saved model
MODEL_SAVE_PATH = 'lstm_model_weights.pth'
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"Model weights saved to {MODEL_SAVE_PATH}")

plt.figure(figsize=(12, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('LSTM Training vs. Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('lstm_loss.png')
print("Saved 'lstm_loss.png'")
plt.close()

print("\n Evaluating on Test Set ")

model.eval() 
with torch.no_grad():
    test_pred_scaled = model(X_test_tensor)
    test_loss_scaled = loss_function(test_pred_scaled, y_test_tensor)
    print(f"Final Test Loss (Scaled MSE): {test_loss_scaled.item():.4f}")
    
    test_pred = scaler.inverse_transform(test_pred_scaled.numpy())
    y_test_orig = scaler.inverse_transform(y_test)
    
    utils.print_metrics(y_test_orig, test_pred, "LSTM")


print(f" STARTING FULL RETRAINING & FORECASTING ")

# We fit the scaler on the entire dataset now
full_numpy = data_series.values
full_scaler = MinMaxScaler(feature_range=(-1, 1))
full_scaled = full_scaler.fit_transform(full_numpy.reshape(-1, 1))

X_full, y_full = create_sequences(full_scaled, look_back)

X_full_tensor = torch.from_numpy(X_full).float()
y_full_tensor = torch.from_numpy(y_full).float()

print(f"Full dataset shape: {X_full_tensor.shape}")

final_model = LSTMModel()
final_model.load_state_dict(torch.load(MODEL_SAVE_PATH))
print(f"Loaded weights from {MODEL_SAVE_PATH}")

optimizer_ft = torch.optim.Adam(final_model.parameters(), lr=0.00005) 
loss_fn = nn.MSELoss()

fine_tune_epochs = 500 
print(f"Fine-tuning on full dataset for {fine_tune_epochs} epochs...")

final_model.train()
for i in range(fine_tune_epochs):
    optimizer_ft.zero_grad()
    y_pred = final_model(X_full_tensor)
    loss = loss_fn(y_pred, y_full_tensor)
    loss.backward()
    optimizer_ft.step()
    
    if (i+1) % 25 == 0:
        print(f"Fine-tune Epoch {i+1}/{fine_tune_epochs} - Loss: {loss.item():.4f}")

print(f"\nGeneratng {future_steps} future predictions...")
final_model.eval()

curr_seq = torch.tensor(full_scaled[-look_back:]).unsqueeze(0).float() # Shape: [1, look_back, 1]
fut_pred_scaled = []

with torch.no_grad():
    for _ in range(future_steps):
        # Predict the next point
        pred = final_model(curr_seq)
        
        # Store prediction
        fut_pred_scaled.append(pred.item())
        
        # Update sequence: Remove first point, add new prediction at the end
        # Shape manipulation to keep dimensions correct for LSTM
        pred_reshaped = pred.unsqueeze(1) # Shape: [1, 1, 1]
        curr_seq = torch.cat((curr_seq[:, 1:, :], pred_reshaped), dim=1)

# Inverse Transform
forecast_values = full_scaler.inverse_transform(np.array(fut_pred_scaled).reshape(-1, 1))

# Saving combined CSV
last_date = data_series.index[-1]
future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), 
                             periods=future_steps, 
                             freq='MS')

df_pred = pd.DataFrame({'DATE': future_dates, 'Forecast_Value': forecast_values.flatten()}).set_index('DATE')
df_full = data_series.rename('Actual_Value').to_frame()
df_full = pd.concat([df_full, df_pred], axis=1)

df_full.to_csv('lstm_full_forecast_and_actuals.csv')
print("Successfully saved combined LSTM data to lstm_full_forecast_and_actuals.csv")

# Plotting the forecast result
output_plot = 'lstm_forecast.png'

plt.figure(figsize=(15, 8))
plt.plot(df_full.index, df_full['Actual_Value'], label='Actual Consumption', color='blue')
plt.plot(df_pred.index, df_pred['Forecast_Value'], label='LSTM Forecast', color='red', linestyle='--')

plt.title('Electricity Consumption: Actuals vs. LSTM Forecast', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.6)
plt.axvline(x=df_pred.index[0], color='green', linestyle=':', linewidth=2, label='Forecast Start')
plt.legend()
plt.tight_layout()
plt.savefig(output_plot)
plt.close()
print(f"Successfully saved plot to {output_plot}")
print(f"Forecast plot saved to '{output_plot}'")

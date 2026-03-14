# Electricity Consumption Forecasting

This repository contains time-series analysis and machine learning models to forecast electricity consumption data up to the year 2024. The original dataset (`27.csv`) was provided by Dr. Samiran Das.

## 📊 Models Implemented

The project explores several forecasting methodologies, including traditional statistical methods and deep learning:

- **SARIMA** (Seasonal AutoRegressive Integrated Moving Average)
- **Prophet** (Facebook Prophet)
- **ETS** (Error, Trend, Seasonal)
- **LSTM** (Long Short-Term Memory Neural Networks)
- **GARCH** (Generalized Autoregressive Conditional Heteroskedasticity)

Models like Prophet and SARIMA also include a 95% confidence interval (upper and lower limits) for their predictions.

## 📁 Project Structure

- `Code/`: Contains all the Python scripts used for analysis and modeling.
  - `analysis.py`, `test_analysis.py`, `stationarity.py`: Exploratory Data Analysis and statistical testing (e.g., Stationarity checks).
  - `FBprophet.py`, `sarima.py`, `ets.py`, `lstm.py`, `garch.py`: Individual model training and forecasting scripts.
  - `plot_compare.py`: Script to generate comparative plots of the forecasted results.
  - `utils.py`: Helper functions.
- `CSV/`: Contains the original dataset (`27.csv`) and the generated out-of-sample forecasts (actuals and predicted values).
- `Plots/`: Output directory where comparative plots and model visual results are saved.

## 🚀 Getting Started

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Vnshiee/Electricity_Consumption_Forecasting.git
   cd Electricity_Consumption_Forecasting
   ```

2. **Run Models**:
   Navigate to the `Code/` directory and run any of the specific models you are interested in. For example:
   ```bash
   python Code/sarima.py
   python Code/lstm.py
   ```

3. **Check Results**:
   Output forecast CSVs will be available in the `CSV/` directory and visual comparisons in the `Plots/` folder.

## 📝 Acknowledgements
Special thanks to Dr. Samiran Das for providing the dataset.

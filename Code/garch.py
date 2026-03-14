import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utils
from arch import arch_model
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import warnings

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
def garch(data_series):
    print("\nSARIMA-GARCH VOLATILITY MODELING\n")
    
    train_data, _ = utils.create_split(data_series)
    print("\n Training Optimal SARIMA Model to get residuals ")
    model = sm.tsa.SARIMAX(train_data.dropna(),
                               order=(2, 1, 3),
                               seasonal_order=(2, 1, 1, 12),
                               enforce_stationarity=False,
                               enforce_invertibility=False).fit(disp=False)
                               
    # get residuals from the SARIMA model and rescale for better GARCH fitting
    residuals = model.resid
    garch_data = residuals * 0.1
    print(f"\nSuccessfully retrieved {len(garch_data)} residuals from SARIMA model.")

    print("\nFitting GARCH(1,1) model on scaled residuals...\n")
    # mean='Zero' assumes the residuals series has a mean of zero
    garch = arch_model(garch_data, vol='Garch', p=1, q=1, mean='Zero', dist='normal')
    garch_results = garch.fit(disp='off')
    
    print("\n GARCH(1,1) Model Summary on SARIMA Residuals (Scaled) ")
    print(garch_results.summary())
    
    # Plot the standardized residuals and conditional volatility
    fig = garch_results.plot(annualize='D')
    fig.set_size_inches(12, 8)
    plt.savefig('garch_sarima_residual_diagnostics.png')
    plt.close() 
    print("\nSaved 'garch_sarima_residual_diagnostics.png'\n")

if __name__ == '__main__':
    data_series = utils.load_data()
    garch(data_series)
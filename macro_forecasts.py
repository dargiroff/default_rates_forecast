"""
Author: Dimitar - d.argiroff@gmail.com
Summary: Forecast the MEVs in the next for the needed periods so that we have 2 years of MEV data after 2020
TODO: The forecasts for long and short interests could be combined into less code
"""
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
from utilities import arima_grid

plt.style.use('bmh')
plt.rcParams['axes.facecolor'] = 'white'

# Each of the arima params takes 0 or 1 as the autoregressive state
p = d = q = range(0, 2)
# Generate all different combinations of p, q and q triplets
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

# FORECAST SHORT INTEREST RATES
df_short_interest = pd.read_csv('data/df_short_interest.csv')
# Check for how many months I need to forecast
periods_short_interest = np.floor((pd.to_datetime('2021-12-01') - max(pd.to_datetime(df_short_interest['time'])))
                                  / np.timedelta64(1, 'M'))
df_short_interest.set_index('time', inplace=True)

# Predict short term interest rates
arima_short_interest = arima_grid(df_short_interest, pdq, seasonal_pdq)
forecast_short_interest = arima_short_interest.iloc[0, 1].get_forecast(steps=int(periods_short_interest))
df_short_interest.reset_index(inplace=True)
fc_values_short_interest = forecast_short_interest.predicted_mean.reset_index()
fc_values_short_interest.columns = ['time', 'value']
df_short_interest = df_short_interest.append(fc_values_short_interest)
df_short_interest = df_short_interest.reset_index(drop=True)


# FORECAST LONG INTEREST RATES
df_long_interest = pd.read_csv('data/df_long_interest.csv')
# Check for how many months I need to forecast
periods_long_interest = np.floor((pd.to_datetime('2021-12-01') - max(pd.to_datetime(df_long_interest['time'])))
                                 / np.timedelta64(1, 'M'))
df_long_interest.set_index('time', inplace=True)

# Predict long term interest rates
arima_long_interest = arima_grid(df_long_interest, pdq, seasonal_pdq)
forecast_long_interest = arima_long_interest.iloc[0, 1].get_forecast(steps=int(periods_long_interest))
df_long_interest.reset_index(inplace=True)
fc_values_long_interest = forecast_long_interest.predicted_mean.reset_index()
fc_values_long_interest.columns = ['time', 'value']
df_long_interest = df_long_interest.append(fc_values_long_interest)
df_long_interest = df_long_interest.reset_index(drop=True)

# FORECAST THE RESAMPLED QUARTERLY VARIABLES
df_quart_resampled = pd.read_csv('data/df_quart_resampled.csv')
# Check for how many months I need to forecast
periods_quart = np.ceil((pd.to_datetime('2021-12-01') - max(pd.to_datetime(df_quart_resampled['time'])))
                        / np.timedelta64(1, 'M'))
df_quart_resampled.set_index('time', inplace=True)

dict_forecasts = dict()
df_forecasted = pd.DataFrame(columns=df_quart_resampled.columns, index=df_long_interest['time'])
for col in df_quart_resampled.columns:
    macro_var = pd.DataFrame(df_quart_resampled[col])
    arima = arima_grid(macro_var, pdq, seasonal_pdq)
    dict_forecasts[col] = arima.iloc[0, 1].get_forecast(steps=int(periods_quart))

    macro_var.reset_index(inplace=True)
    fc_values = dict_forecasts[col].predicted_mean.reset_index()
    fc_values.columns = ['time', 'value']
    macro_var.columns = ['time', 'value']
    macro_var = macro_var.append(fc_values)
    macro_var = macro_var.reset_index(drop=True)
    df_forecasted[col] = macro_var['value'].values

# Combine all forecasted MEVs into a single dataset
df_forecasted['long_interest'] = df_long_interest['value'].values
df_forecasted['short_interest'] = df_long_interest['value'].values
df_forecasted['date'] = df_long_interest['time'].values

# Save the final MEVs dataset to be used in modelling
df_forecasted.to_csv('data/df_forecasted_mevs.csv', index=False)

# Plot the MEV forecasts
df_forecasted.index = pd.to_datetime(df_forecasted.index)
ax = df_forecasted.loc[:, ['gdp_growth', 'imports', 'exports',
                           'unemployment', 'short_interest', 'long_interest',
                           'production', 'labor_cost_index']].plot()
plt.axvline(pd.to_datetime('2020-06-01'), linestyle='--', color='red', label='Prediction Start')

plt.title('MEV Timeseries')
plt.xlabel('Date')
plt.ylabel('Value')

plt.legend(loc='upper left')
plt.xticks(rotation=-45)
plt.show()

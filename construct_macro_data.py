"""
Author: Dimitar - d.argiroff@gmail.com
Summary: Constructs datasets with MEVs
TODO:
"""
import pandas as pd
import numpy as np

# GPD data
df_gdp_all = pd.read_csv('data/oecd_data/gdp_quarterly.csv')
df_gdp_all['TIME'] = pd.to_datetime(df_gdp_all['TIME'])

df_gdp = df_gdp_all[df_gdp_all['SUBJECT'] == 'B1_GE']
gdp_growth = df_gdp.Value
time = df_gdp.TIME
df_exports = df_gdp_all[df_gdp_all['SUBJECT'] == 'P6']
exports = df_exports.Value
df_imports = df_gdp_all[df_gdp_all['SUBJECT'] == 'P7']
imports = df_imports.Value
df_capital_formation = df_gdp_all[df_gdp_all['SUBJECT'] == 'P51']
capital_formation = df_capital_formation.Value

# Unemployment data
df_unemployment_all = pd.read_csv('data/oecd_data/unemployment_quarterly.csv')
# Reformat the quarterly data so I can cast it into datetime type
df_unemployment_all['year'] = [date[3:] for date in df_unemployment_all['Time']]
df_unemployment_all['quarter'] = [date[:2] for date in df_unemployment_all['Time']]
df_unemployment_all['TIME'] = df_unemployment_all.year + '-' + df_unemployment_all.quarter
df_unemployment_all['TIME'] = pd.to_datetime(df_unemployment_all['TIME'])

# Fill in one missing umeployment number by appending the average of the unemployment for the last 3 quarters
unemployment = np.array(df_unemployment_all.Value)

# House price data
df_houseprice_all = pd.read_csv('data/oecd_data/house_price_index_quarterly.csv')
df_houseprice_all['TIME'] = pd.to_datetime(df_houseprice_all['TIME'])
df_houseprice = df_houseprice_all[df_houseprice_all['MEASURE'] == 'IXOB']
# House prices data is available for one quarter less - assing the missing quarter value to be equal to the average
# of the previous two quarters
house_price_index = df_houseprice.Value
house_price_index = house_price_index.append(pd.Series((house_price_index.iloc[-1] + house_price_index.iloc[-2]) / 2.,
                                                       index=[house_price_index.index[-1] + 1]))

# Interest rates
df_interest = pd.read_csv('data/oecd_data/interests_monthly.csv')
df_interest['TIME'] = pd.to_datetime(df_interest['TIME'])
df_long_interest = df_interest[df_interest['SUBJECT'] == 'IRLT'][['TIME', 'Value']]
df_short_interest = df_interest[df_interest['SUBJECT'] == 'IR3TIB'][['TIME', 'Value']]

# Unit labor costs
df_laborcosts = pd.read_csv('data/oecd_data/labour_cost_quarterly.csv')
df_laborcosts['TIME'] = pd.to_datetime(df_laborcosts['TIME'])
laborcosts_index = df_laborcosts['Value']

# Production
df_production = pd.read_csv('data/oecd_data/industrial_prod_quarterly.csv')
df_production = df_production.loc[:44]
production = df_production['Value']

df_quarterly_macro = pd.DataFrame({'time': time,
                                   'gdp_growth': np.array(gdp_growth),
                                   'exports': np.array(exports),
                                   'imports': np.array(imports),
                                   'capital_formation': np.array(capital_formation),
                                   'unemployment': np.array(unemployment),
                                   'house_price_index': np.array(house_price_index),
                                   'labor_costs_index': np.array(laborcosts_index),
                                   'production': np.array(production)})

# Resample the quarterly data into monthly
# Linear interpolation is used
df_quart_resampled = df_quarterly_macro.resample('M', on='time', convention='start').mean().interpolate().reset_index()
df_quart_resampled['time'] = df_quart_resampled['time'].apply(lambda dt: dt.replace(day=1))

# Save the macro variables datasets - to be used in future prediction via a SARIMAX model
df_quarterly_macro.columns = [col.lower() for col in df_quarterly_macro.columns]
df_production.columns = [col.lower() for col in df_production.columns]
df_long_interest.columns = [col.lower() for col in df_long_interest.columns]
df_short_interest.columns = [col.lower() for col in df_short_interest.columns]

df_quart_resampled.to_csv('data/df_quart_resampled.csv', index=False)
df_production.to_csv('data/df_production.csv', index=False)
df_long_interest.to_csv('data/df_long_interest.csv', index=False)
df_short_interest.to_csv('data/df_short_interest.csv', index=False)


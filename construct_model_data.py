"""
Author: Dimitar - d.argiroff@gmail.com
Summary: Explore the default rates data, convert variable format and names, merge the default data with the MEVs data
TODO:
"""
import pandas as pd
from utilities import summarize_missings, vif_variable_selection, calc_correlation, plot_feature_target_scatter, plot_dr
from matplotlib import pyplot as plt

plt.style.use('bmh')
plt.rcParams['axes.facecolor'] = 'white'

# Load in the defaults dataset
df_defaults = pd.read_csv('data/test_data_v20200706.txt', delimiter='|')
df_defaults.columns = [col.lower() for col in df_defaults]
df_defaults.rename(columns={'pd': 'default_rate'}, inplace=True)
df_defaults['date'] = pd.to_datetime(df_defaults['date'])

# Load in the macro variables dataset
df_macro = pd.read_csv('data/df_forecasted_mevs.csv')
df_macro['date'] = pd.to_datetime(df_macro['date'])
df_know_dr_macro = df_macro[df_macro['date'] < pd.to_datetime('2020-01-01')]
df_unknown_dr_macro = df_macro[df_macro['date'] >= pd.to_datetime('2020-01-01')]

# Check features correlation
df_corr, df_pvals = calc_correlation(df=df_know_dr_macro.iloc[:, :-1], plot=True)

# Eliminate features based on variance inflation factor > 5.
df_subsel_kn_macro = vif_variable_selection(X=df_know_dr_macro.iloc[:, :-1], thresh=5.)
df_subsel_ukn_macro = df_unknown_dr_macro[df_subsel_kn_macro.columns]

df_subsel_kn_macro.loc[:, 'date'] = df_know_dr_macro['date']
df_subsel_ukn_macro.loc[:, 'date'] = df_unknown_dr_macro['date']

# Create a sample with known default rates and sample with the unknown default rates (2020 and 2021)
df_known_dr = pd.merge(df_defaults, df_subsel_kn_macro, on='date')

# Inspect the relationship between the features and the target
plot_feature_target_scatter(*df_known_dr.columns[6:12], target='default_rate', data=df_known_dr)

# Check for missing values
nans_summary = summarize_missings(df=df_known_dr, subset=df_known_dr.columns)

# Plot the default rate over time
fig, ax = plt.subplots()
ax.plot(df_known_dr['date'], df_known_dr['default_rate'] * 100., label='Default Rate %')
ax.set_title('Default Rate Over Time')
ax.set_xlabel('Date')
ax.set_ylabel('Default Rate %')
plt.legend()
plt.show()

# Save the modelling datasets to use further
df_known_dr.to_csv('data/modelling_dataset.csv', index=False)
df_subsel_ukn_macro.to_csv('data/future_features.csv', index=False)





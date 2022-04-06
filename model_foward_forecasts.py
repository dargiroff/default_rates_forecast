"""
Author: Dimitar - d.argiroff@gmail.com
Summary: Does a forward looking forecast using all available data
TODO:
"""
import pandas as pd
from sklearn.linear_model import LinearRegression
from utilities import plot_dr, report_feature_importance

# Load the data and split it into training and testing sets
future_features = pd.read_csv('data/future_features.csv')
data_model = pd.read_csv('data/modelling_dataset.csv')
X_train, y_train = data_model.iloc[:, 6:], data_model['default_rate']
X_test = future_features.iloc[:, :-1]

# Fit the model
model = LinearRegression()
model.fit(X=X_train, y=y_train)
# Make a 12 month forecast
y_pred_12 = model.predict(X=X_test[:12])
df_y_pred_12 = pd.DataFrame(index=pd.to_datetime(future_features['date'][:12]))
df_y_pred_12['pred'] = y_pred_12
df_y_train = pd.DataFrame(index=pd.to_datetime(data_model['date']))
df_y_train['true'] = y_train.values

# Connect the true values and the prediction
fill = pd.DataFrame(index=[pd.to_datetime('2019-12-01')], columns={'pred': []})
fill = fill.fillna(df_y_train.iloc[-1].values[0])
df_y_pred_12 = pd.concat([fill, df_y_pred_12])

# Plot the 12 month forecast
plot_dr(df_y_train, df_y_pred_12, n_xticks=10)

# Make a 24 month forecast
y_pred_24 = model.predict(X=X_test[:24])
df_y_pred_24 = pd.DataFrame(index=pd.to_datetime(future_features['date'][:24]))
df_y_pred_24['pred'] = y_pred_24

# Connect the true values and the prediction
df_y_pred_24 = pd.concat([fill, df_y_pred_24])

# Plot the 24 month forecast
plot_dr(df_y_train, df_y_pred_24, n_xticks=10)

# Evaluate feature importance
report_feature_importance(coefficients=model.coef_, labels=X_train.columns)

"""
Author: Dimitar - d.argiroff@gmail.com
Summary: Backtests a Linear Model in the sample, where the default rates are known
TODO:
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
import statsmodels.api as sm
import matplotlib.pyplot as plt

from utilities import evaluate_model

# Load in the dataset and split into training and testing samples
data_model = pd.read_csv('data/modelling_dataset.csv')
X, y = data_model.iloc[:, 6:], data_model['default_rate']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.66, random_state=42)

# Fit the model and make a prediction
model = LinearRegression()
res = model.fit(X=X_train, y=y_train)
y_pred = model.predict(X=X_test)

# Evaluate the model prediction
mse = evaluate_model(y_test=y_test, y_pred=y_pred, metric='mse')
r2 = evaluate_model(y_test=y_test, y_pred=y_pred, metric='r2')
comp = pd.DataFrame({'true': y_test.values,
                     'pred': y_pred})

# Cross validated evaluation
cv_mses = -1. * cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', cv=10)
cv_r2s = cross_val_score(model, X_train, y_train, scoring='r2', cv=10)

average_r2 = np.average(cv_r2s)
average_mse = np.average(cv_mses)

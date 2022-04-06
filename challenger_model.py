"""
Author: Dimitar - d.argiroff@gmail.com
Summary: Tries to improve on the main model performance via alternative techniques - LGBM
TODO:
"""
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split, ParameterGrid

from utilities import evaluate_model

# Fit the model and make a prediction
data_model = pd.read_csv('data/modelling_dataset.csv')
X, y = data_model.iloc[:, 6:], data_model['default_rate']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.66, random_state=42)

# Create an optimized dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

# Set up a dict of possible hyperparams
params = {
    'objective': ['regression'],
    'num_iterations': [1000],
    'learning_rate': [0.1, 0.05, 0.02, 0.01],
    'feature_fraction': [1., 0.08],
    'metric': ['l2', 'l1'],
    'num_leaves': [leaf for leaf in range(3, 21)],
}

param_combos = list(ParameterGrid(params))

ls_r2 = list()
ls_mse = list()
ls_models = list()
for param_set in param_combos:
    # Fit the model and make a prediction
    model = lgb.train(param_set,
                      lgb_train,
                      valid_sets=lgb_eval)
    y_pred = model.predict(X_test)

    # Evaluate the model prediction
    ls_mse.append(evaluate_model(y_test=y_test, y_pred=y_pred, metric='mse'))
    ls_r2.append(evaluate_model(y_test=y_test, y_pred=y_pred, metric='r2'))
    ls_models.append(model)

# Check the best results
df_results = pd.DataFrame({'model': ls_models,
                           'r2': ls_r2,
                           'mse': ls_mse})
df_results.sort_values(by=['r2', 'mse'], ascending=[False, True], inplace=True)
df_results.reset_index(drop='index', inplace=True)

best_r2 = df_results.loc[0, 'r2']
best_mse = df_results.loc[0, 'mse']

"""
Author: Dimitar - d.argiroff@gmail.com
Summary: General functions that are used across scripts
TODO: Add docstrings to the functions
"""
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, roc_auc_score, accuracy_score, r2_score
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from matplotlib import pyplot as plt
import seaborn as sns

plt.style.use('bmh')
plt.rcParams['axes.facecolor'] = 'white'


def summarize_missings(df, subset):
    df = df[subset]
    percent = 100 * (len(df) - df.count()) / len(df)
    total = (len(df) - df.count())
    out = pd.concat([total, percent], axis=1)
    out.columns = ["Count", "Percent"]

    return out


def evaluate_model(y_test, y_pred, metric):
    if metric == 'mse':
        return mean_squared_error(y_test, y_pred)
    elif metric == 'rmse':
        return np.sqrt(mean_squared_error(y_test, y_pred))
    elif metric == 'auc':
        return roc_auc_score(y_test, y_pred)
    elif metric == 'accuracy':
        return accuracy_score(y_test, y_pred)
    elif metric == 'r2':
        return r2_score(y_test, y_pred)

    else:
        return ValueError('\nThe metric must be in [\'mse\'; \'rmse\'; \'auc\'; \'accuracy\']')


def arima_grid(series, pdq, seasonal_pdq):
    ls_results = list()
    ls_models = list()
    ls_aics = list()
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            model = sm.tsa.statespace.SARIMAX(series,
                                              order=param,
                                              seasonal_order=param_seasonal,
                                              enforce_stationarity=False,
                                              enforce_invertibility=False)
            ls_models.append(model)
            result = model.fit()
            ls_results.append(result)
            ls_aics.append(result.aic)

    df_results = pd.DataFrame({'model': ls_models,
                               'result': ls_results,
                               'aic_evaluation': ls_aics})
    # Lowest AIC (highest abs value) on top - best model
    df_results.sort_values(by='aic_evaluation', inplace=True)

    return df_results


def vif_variable_selection(X, thresh=5.0):
    variables = list(range(X.shape[1]))
    dropped = True
    while dropped:
        dropped = False
        vif = [variance_inflation_factor(X.iloc[:, variables].values, ix)
               for ix in range(X.iloc[:, variables].shape[1])]

        print(vif)
        maxloc = vif.index(max(vif))
        if max(vif) > thresh:
            print('dropping \'' + X.iloc[:, variables].columns[maxloc] +
                  '\' at index: ' + str(maxloc))
            del variables[maxloc]
            dropped = True

    print('Remaining variables:')
    print(X.columns[variables])

    return X.iloc[:, variables]


def plot_dr(true, pred, n_xticks=3):
    fig, ax = plt.subplots()

    ax.plot(true * 100., label='True Default Rate %')
    ax.plot(pred * 100., label='Default Rate Prediction %')

    ax.set_xlabel('Date')
    ax.set_ylabel('Default Rate %')

    ax.xaxis.set_major_locator(plt.MaxNLocator(n_xticks))
    plt.xticks(rotation=-45)
    plt.legend()
    plt.show()


def report_feature_importance(coefficients, labels):
    for idx, coeff in enumerate(coefficients):
        print(f'Feature: {labels[idx]} Score: {coeff}')

    plt.bar([col for col in labels], coefficients)
    plt.xticks(rotation=-45)
    plt.xlabel('Feature')
    plt.ylabel('Coefficient')
    plt.show()


def calc_correlation(df, plot=False):
    corr = df.corr()
    pvals = df.corr(method=lambda x, y: pearsonr(x, y)[1]) - np.eye(len(df.columns))

    if plot:
        sns.heatmap(corr, annot=False, cmap=plt.cm.Reds)
        plt.show()

    for i, row in enumerate(corr.columns):
        for j, column in enumerate(corr.columns):
            if abs(pvals.iloc[i][column]) < 0.01:
                corr.loc[row, column] = str(np.round(corr.loc[row, column], 4)) + '***'
            elif abs(pvals.iloc[i][column]) < 0.05:
                corr.loc[row, column] = str(np.round(corr.loc[row, column], 4)) + '**'
            elif abs(pvals.iloc[i][column]) < 0.1:
                corr.loc[row, column] = str(np.round(corr.loc[row, column], 4)) + '*'
            else:
                corr.loc[row, column] = str(np.round(corr.loc[row, column], 4))

    return corr, pvals


def plot_feature_target_scatter(*features, target, data):
    fig, axs = plt.subplots(len(features), figsize=(8, 2 * len(features)))
    for idx in range(len(features)):
        axs[idx].scatter(data[features[idx]], data[target], s=10)
        axs[idx].set_title(str(target) + ' vs ' + str(features[idx]))
        axs[idx].set_facecolor('white')

    for idx, ax in enumerate(axs.flat):
        ax.set(xlabel=features[idx], ylabel=target)

    plt.tight_layout()
    plt.show()

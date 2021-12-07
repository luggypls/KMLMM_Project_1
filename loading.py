import pandas as pd
import numpy as np


def lognuniform(low=0, high=1, size=None, base=10):
    return np.power(base, np.random.uniform(low, high, size))


def _nan_handler1(data: pd.DataFrame)-> pd.DataFrame:
    for row in data.index:
        if data.loc[row,:].isna().sum() > 0:
            nan_columns = data.loc[row, :].isna()
            ids = data.iloc[:,0]
            data.loc[row, nan_columns] = data.loc[ids==ids[row], nan_columns].mean()
        else:
            pass
    return data


def _nan_handler2(data: pd.DataFrame)-> pd.DataFrame:
    data = data.dropna(axis=0)
    return data


def _outlier_handler(data: pd.DataFrame)-> pd.DataFrame:
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    lower = mean - 3 * std
    upper = mean + 3 * std
    data = np.clip(data, lower, upper, axis=1)
    return data


def load_data(path: str)-> pd.DataFrame:
    data = pd.read_csv(path)
    data = _nan_handler2(data)
    data = _outlier_handler(data)
    target = data.loc[:,'class']
    X = data.drop(['id', 'class'], axis=1)
    return X, target

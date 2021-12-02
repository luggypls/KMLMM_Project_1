import pandas as pd
import numpy as np


def _nan_handler1(data: pd.DataFrame)-> pd.DataFrame:
    for row in data.index:
        if data.loc[row,:].isna().sum() > 0:
            nan_columns=data.loc[row,:].isna()
            ids=data.iloc[:,0]
            data.loc[row, nan_columns] = data.loc[ids==ids[row], nan_columns].mean()
        else:
            pass
    return data


def _nan_handler2(data: pd.DataFrame)-> pd.DataFrame:
    for row in data.index:
        if data.loc[row,:].isna().sum() > 0:
            data=data.drop(row, axis=0)
        else:
            pass
        return data


def _outlier_handler(data: pd.DataFrame)-> pd.DataFrame:
    mean=data.mean(axis=0)
    std=data.std(axis=0)
    lower=mean+3*std
    upper=mean-3*std
    for col in data.columns:
        data[col]=np.where(data[col]<lower[col], lower[col], data[col])
        data[col]=np.where(data[col]<upper[col], upper[col], data[col])
    return data


def load_data(path: str)-> pd.DataFrame:
    data = pd.read_csv(path)
    data = _nan_handler2(data)
    data = _outlier_handler(data)
    target = data.loc[:,'class']
    X = data.drop(['id', 'class'], axis=1)
    return X, target



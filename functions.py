


import pandas as pd
import numpy as np


data_path='./Data/pd_speech_features.csv'

def _nan_handler(data: pd.DataFrame)-> pd.DataFrame:
    for row in data.index:
        if data.loc[row,:].isna().sum() > 0:
            nan_columns=data.loc[row,:].isna()
            ids=data.iloc[:,0]
            data.loc[row, nan_columns] = data.loc[ids==ids[row], nan_columns].mean()
        else:
            pass
    return data


def _z(col: pd.Series) -> pd.DataFrame:
    med_col = col.mean()
    med_abs_dev = (np.abs(col - med_col)).median()
    mod_z = ((col - med_col) / med_abs_dev)
    return np.abs(mod_z)


def _mod_z(col: pd.Series) -> pd.DataFrame:
    med_col = col.median()
    med_abs_dev = (np.abs(col - med_col)).median()
    mod_z = 0.6745 * ((col - med_col) / med_abs_dev)
    return np.abs(mod_z)


def _outlier_handler(data: pd.DataFrame)-> pd.DataFrame:
    z_score = data.apply(_z)
    return z_score


def load_data(path: str)-> pd.DataFrame:
    data = pd.read_csv(path)
    no_nans = _nan_handler(data)
    target = no_nans.loc[:,'class']
    X = data.drop(['id', 'class'], axis=1)
    return X, target

    

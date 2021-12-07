import pandas as pd
import numpy as np
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split


def split_and_scale(X: pd.DataFrame, y: pd.Series)-> pd.DataFrame:
    X_train, X_test, y_train, y_test = train_test_split(X,y,stratify=y)
    scaling = MaxAbsScaler().fit(X_train)
    X_train = pd.DataFrame(scaling.transform(X_train), index=X_train.index, columns=X_train.columns)
    X_test =  pd.DataFrame(scaling.transform(X_test), index=X_test.index, columns=X_test.columns)
    return X_train, X_test, y_train, y_test
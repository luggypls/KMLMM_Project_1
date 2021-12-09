from sklearn.preprocessing import MaxAbsScaler

def scale(X: pd.DataFrame)-> pd.DataFrame:
    return MaxAbsScaler(X)
import pandas as pd
from sklearn.decomposition import KernelPCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from skopt import BayesSearchCV

class GridKernelPCA():
    def __init__(self,
                 data: pd.DataFrame,
                 model_params: dict
                 ):
        self.__gridsearch = self.__make_grid_search(data,model_params)
        self._cv_results=self.__gridsearch.cv_results_
        self.__results=self.__make_results()

    @staticmethod
    def __my_scorer(estimator, X):
        X_reduced = estimator.transform(X)
        X_preimage = estimator.inverse_transform(X_reduced)
        return -1 * mean_squared_error(X, X_preimage)

    def __make_grid_search(self, data, model_params)-> GridSearchCV:
        kpca=KernelPCA(fit_inverse_transform=True, n_jobs=-1)
        gs = GridSearchCV(estimator=kpca,
                            scoring=self.__my_scorer,
                            param_grid=model_params,
                            cv=3
                            )
        gs.fit(data)
        return gs
    
    def __make_results(self)-> pd.DataFrame:
        return pd.DataFrame(self.__gridsearch.cv_results_).sort_values(
            by='mean_test_score', ascending=False)['params', 'mean_test_score']
    
    def get_best_params(self)-> dict:
        return self.__gridsearch.best_params_
    
    def get_results(self)-> pd.DataFrame:
        return self.__results
    
    
class BayesKernelPCA():
    def __init__(self,
                 data: pd.DataFrame,
                 model_params: dict,
                 n_iter: int,
                 kernel: str,
                 ):
        self.__gridsearch = self.__make_grid_search(kernel,data,model_params, n_iter)
        self._cv_results=self.__gridsearch.cv_results_
        self.__results=self.__make_results()

    @staticmethod
    def __my_scorer(estimator, X):
        X_reduced = estimator.transform(X)
        X_preimage = estimator.inverse_transform(X_reduced)
        return -1 * mean_squared_error(X, X_preimage)

    def __make_grid_search(self,kernel, data, model_params, n_iter)-> GridSearchCV:
        kpca=KernelPCA(kernel=kernel, fit_inverse_transform=True, n_jobs=-1)
        gs = BayesSearchCV(estimator=kpca,
                            scoring=self.__my_scorer,
                            search_spaces=model_params,
                            cv=3,
                            n_iter=n_iter
                            )
        gs.fit(data)
        return gs
    
    def __make_results(self)-> pd.DataFrame:
        return pd.DataFrame(self.__gridsearch.cv_results_).sort_values(
            by='mean_test_score', ascending=False).loc[:,['params', 'mean_test_score']]
    
    def get_best_params(self)-> dict:
        return self.__gridsearch.best_params_
    
    def get_results(self)-> pd.DataFrame:
        return self.__results    
    

#todo add transform
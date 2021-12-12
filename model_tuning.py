import pandas as pd
from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV

class GridSearch():
    def __init__(self,
                 data: pd.DataFrame,
                 target: pd.Series,
                 model: object,
                 metrics: dict,
                 model_params: dict
                 ):
        self.__scoring_cols = self.__make_scoring_cols(metrics)
        self.__gridsearch = self.__make_grid_search(data,target,model,
                                                  metrics,model_params)
        self._cv_results=self.__gridsearch.cv_results_
        self.__results=self.__make_results()

    def __make_grid_search(self, data, target, model, metrics, model_params
                           )-> GridSearchCV:
        gs = GridSearchCV(estimator=model,
                            scoring=metrics,
                            param_grid=model_params,
                            n_jobs=-1,
                            refit=self.__scoring_cols[0].replace('mean_test_','')
                            )
        gs.fit(data, target)
        return gs
        
    def __make_scoring_cols(self, metrics)-> list:
        tmp=[]
        cte='mean_test_'
        for metric in metrics:
            tmp.append(cte+metric)
        return tmp
    
    def __make_results(self)-> pd.DataFrame:
        return pd.DataFrame(self.__gridsearch.cv_results_).sort_values(
            by=self.__scoring_cols[0], ascending=False)[self.__scoring_cols]
    
    def get_best_params(self)-> dict:
        return self.__gridsearch.best_params_
    
    def get_results(self)-> pd.DataFrame:
        return self.__results


class BayesSearch():
    def __init__(self,
                 data: pd.DataFrame,
                 target: pd.Series,
                 model: object,
                 metrics: dict,
                 model_params: dict,
                 n_iter: int,
                 ):
        self.__gridsearch = self.__make_grid_search(data,target,model,
                                                  metrics,model_params,n_iter)
        self._cv_results=self.__gridsearch.cv_results_
        self.__results=self.__make_results()

    def __make_grid_search(self, data, target, model, metrics, model_params, n_iter
                           )-> BayesSearchCV:
        gs = BayesSearchCV(estimator=model,
                            scoring=metrics,
                            search_spaces=model_params,
                            n_jobs=-1,
                            n_iter=n_iter,
                            )
        gs.fit(data, target)
        return gs
    
    def __make_results(self)-> pd.DataFrame:
        return pd.DataFrame(self.__gridsearch.cv_results_).sort_values(
            by='mean_test_score', ascending=False).loc[:,['params', 'mean_test_score']]
    
    def get_best_params(self)-> dict:
        return self.__gridsearch.best_params_
    
    def get_results(self)-> pd.DataFrame:
        return self.__results
    
from sklearn.model_selection import GridSearchCV
from kfdaM import Kfda
import pandas as pd

class GridKernelFDA():
    def __init__(self,
                 data: pd.DataFrame,
                 target: pd.Series,
                 model_params: dict,
                 metrics: dict
                 ):
        self.__scoring_cols = self.__make_scoring_cols(metrics)
        self.__gridsearch = self.__make_grid_search(data,target,
                                                  metrics,model_params)
        self._cv_results=self.__gridsearch.cv_results_

    def __make_grid_search(self, data, target, metrics, model_params
                           )-> GridSearchCV:
        model = Kfda()
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
    
    def get_best_params(self)-> dict:
        return self.__gridsearch.best_params_
    
    def get_best_score(self)-> float:
        return self.__gridsearch.best_score_
    
    def get_results(self)-> pd.DataFrame:
        return self._cv_results

import loading as l
import gridsearch as g

from sklearn.svm import SVC
from sklearn.metrics import make_scorer, roc_auc_score, recall_score

data_path='./Data/pd_speech_features.csv'

X,y = l.load_data(data_path)

roc_scorer = make_scorer(roc_auc_score)
recall_scorer = make_scorer(recall_score)

metrics = {'accuracy': 'accuracy',''
           'roc': roc_scorer,
           'recall': recall_scorer
           }

model_params = {'kernel' : ['linear', 'rbf', 'poly'],
                'C' : [1, 10],
                'degree' : [2, 3, 10],
                'gamma' : ['scale', 'auto']
                }

clf=SVC()
out=g.GridSearch(X,y,clf, metrics, model_params)
print(out.get_results)
import loading as l
import model_tuning as g
import kernelpca_gs as k
import scaling as s

import numpy as np
import pandas as pd
from sklearn.svm import SVC,LinearSVC
from sklearn.decomposition import KernelPCA
from sklearn.metrics import make_scorer, roc_auc_score, recall_score, accuracy_score,\
confusion_matrix, precision_score, classification_report
from sklearn.model_selection import train_test_split

data_path='./Data/pd_speech_features.csv'
kernel_params_path='./Params/Kpca_params.csv'
model_params_path='./Params/rbf_params.csv'

X,y = l.load_data(data_path)

X_train, X_test, y_train, y_test = s.split_and_scale(X, y)


params={'alpha': l.lognuniform(low=-4,high=2,size=40, base=10),
        'gamma': l.lognuniform(low=-4,high=2,size=40, base=10)
        }

kpca_bs=k.BayesKernelPCA(X_train, params, n_iter=200, kernel='rbf')

alpha=kpca_bs.get_best_params()['alpha']
gamma=kpca_bs.get_best_params()['gamma']
score=kpca_bs.get_results().iloc[0,-1]

tmp=pd.DataFrame({'alpha':alpha, 'gamma':gamma, 'score':score}, index=['rbf'])

tmp.to_csv(kernel_params_path)




#kpca=KernelPCA(alpha=0.0001, gamma=0.005, kernel='rbf').fit(X_train)
kpca=KernelPCA(alpha=alpha, gamma=gamma, kernel='rbf').fit(X_train)

print('kpca done!')



X_train=kpca.transform(X_train)#[:,rel_dim]
X_test=kpca.transform(X_test)#[:,rel_dim]




params={'C': l.lognuniform(low=-3,high=2,size=60, base=10),
        'gamma': l.lognuniform(low=-4,high=2,size=60, base=10)
        }

metrics=make_scorer(roc_auc_score)
clf=SVC(kernel='rbf')
out=g.BayesSearch(X_train, y_train, clf, metrics, params, n_iter=200)

rbf_best_params=out.get_best_params()

#clf=SVC(kernel='rbf', C=27.43, gamma=2.09)
clf=SVC(kernel='rbf',**rbf_best_params)
clf.fit(X_train, y_train)
print(confusion_matrix(y_test, clf.predict(X_test)))
print(classification_report(y_test, clf.predict(X_test)))

rbf_best_params.update({'score':out.get_results().iloc[0,-1]})

tmp=pd.DataFrame(rbf_best_params, index=['rbf'])

print('rbf done!')

poly_params={'C': l.lognuniform(low=-3,high=2,size=60, base=10),
        'gamma': l.lognuniform(low=-4,high=2,size=60, base=10),
        'degree': [2,3,4,5,6,7,8,9,10],
        'coef0': l.lognuniform(low=-4,high=2,size=60, base=10)
        }
clf=SVC(kernel='poly')
out=g.BayesSearch(X_train, y_train, clf, metrics, poly_params, n_iter=200)
poly_best_params=out.get_best_params()
clf=SVC(kernel='poly',**poly_best_params)
clf.fit(X_train, y_train)
print(confusion_matrix(y_test, clf.predict(X_test)))
print(classification_report(y_test, clf.predict(X_test)))
poly_best_params.update({'score':out.get_results().iloc[0,-1]})
series=pd.Series(poly_best_params, name='poly')

tmp.append(series)



tmp.to_csv(model_params_path)

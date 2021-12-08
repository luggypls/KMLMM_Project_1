import loading as l
import model_tuning as g
import kernelpca_gs as k
import scaling as s

import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.svm import SVC,LinearSVC
from sklearn.decomposition import KernelPCA
from sklearn.metrics import make_scorer, roc_auc_score, recall_score, accuracy_score,\
confusion_matrix, precision_score, classification_report, balanced_accuracy_score
from sklearn.model_selection import train_test_split

data_path='./Data/pd_speech_features.csv'
kernel_params_path='./Params/Kpca_params.csv'
model_params_path='./Params/rbf_params.csv'

X,y = l.load_data(data_path)

X_train, X_test, y_train, y_test = s.split_and_scale(X, y)


params={'alpha': l.lognuniform(low=-4,high=2,size=50, base=10),
        'gamma': l.lognuniform(low=-4,high=2,size=50, base=10)
        }

kpca_bs=k.BayesKernelPCA(X_train, params, n_iter=200, kernel='rbf')

alpha=kpca_bs.get_best_params()['alpha']
gamma=kpca_bs.get_best_params()['gamma']
score=kpca_bs.get_results().iloc[0,-1]


alpha=_best_params['alpha']
gamma=_best_params()['gamma']
score=kpca_bs.get_results().iloc[0,-1]
tmp=pd.DataFrame({'alpha':alpha, 'gamma':gamma, 'score':score}, index=['rbf'])

#tmp.to_csv(kernel_params_path)




#kpca=KernelPCA(alpha=0.0001, gamma=0.005, kernel='rbf').fit(X_train)
kpca=KernelPCA(alpha=alpha, gamma=gamma, kernel='rbf').fit(X_train)

print('kpca done!')



X_train=kpca.transform(X_train)[:,:110]
X_test=kpca.transform(X_test)[:,:110]




params={'C': l.lognuniform(low=-3,high=2,size=60, base=10),
        'gamma': l.lognuniform(low=-4,high=2,size=60, base=10)
        }

metrics=make_scorer(roc_auc_score)
clf=SVC(kernel='rbf')
out1=g.BayesSearch(X_train, y_train, clf, metrics, params, n_iter=250)

rbf_best_params2=out1.get_best_params()

#clf=SVC(kernel='rbf', C=27.43, gamma=2.09)
clf=SVC(kernel='rbf',**rbf_best_params2)
clf.fit(X_train, y_train)
print(confusion_matrix(y_test, clf.predict(X_test)))
print(classification_report(y_test, clf.predict(X_test)))

rbf_best_params.update({'score':out1.get_results().iloc[0,-1]})

tmp=pd.DataFrame(rbf_best_params2, index=['rbf'])

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

tmp=tmp.append(series)

p=sns.lineplot(x=range(len(kpca.lambdas_)),y=np.cumsum(kpca.lambdas_)/np.sum(kpca.lambdas_))
p.set_xlabel("# PC")
p.set_ylabel("% Explained Variance")



[[109  83]
 [ 12 552]]

[[106  86]
 [ 12 552]]   #rbf_best_params all

[[124  68]
 [ 14 550]]   #+rbf_best_params X[:300]

[[129  63]
 [ 21 543]]   #+rbf_best_params X[:200]

[[138  54]
 [ 21 543]]   #rbf_best_params X[:150]

[[140  52]
 [ 23 541]]   #rbf_best_params X[:110]

#tmp.to_csv(model_params_path)

from sklearn.model_selection import LeaveOneOut

from sklearn.preprocessing import MaxAbsScaler
scaling = MaxAbsScaler().fit(X)
X2 = pd.DataFrame(scaling.transform(X), index=X.index, columns=X.columns)
loo = LeaveOneOut()
loo.get_n_splits(X2)

params={'alpha': l.lognuniform(low=-4,high=2,size=50, base=10),
        'gamma': l.lognuniform(low=-4,high=2,size=50, base=10)
        }

kpca_bs=k.BayesKernelPCA(X2, params, n_iter=100, kernel='rbf')
_best_params=kpca_bs.get_best_params()
kpca=KernelPCA(**_best_params, kernel='rbf').fit(X2)

scores2=[]   
sizes=[5,25,50,75,100,110,130,175,200,250,300,400,500,600,700,750]

for i in sizes:
    X3=kpca.transform(X2)[:,:i]
    
    y_pred=[]
    for train_index, test_index in loo.split(X3):
        X_train, X_test = X3[train_index], X3[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf=SVC(kernel='rbf',**rbf_best_params2)
        clf.fit(X_train, y_train)
        y_pred.append(clf.predict(X_test))
        
    scores2.append(roc_auc_score(y, y_pred))

plt.plot(sizes,scores2)



p=sns.lineplot(x=sizes,y=scores2)
p.set_xlabel("# PC")
p.set_ylabel("ROC score")


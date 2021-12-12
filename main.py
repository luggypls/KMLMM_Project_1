import loading as l
import model_tuning as g
import kernelpca_gs as k
import scaling as s
from kfdaM import Kfda
from GridKernelFDA import GridKernelFDA
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import numpy as np
import pandas as pd
from sklearn.svm import SVC,LinearSVC

from sklearn.decomposition import KernelPCA
from sklearn.metrics import make_scorer, roc_auc_score, recall_score, accuracy_score,\
confusion_matrix, precision_score, classification_report, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split

data_path='./Data/pd_speech_features.csv'
kernel_params_path='./Params/Kpca_params.csv'
model_params_path='./Params/model_params.csv'


X, y = l.load_data(data_path)
X_train, X_test, y_train, y_test = s.split_and_scale(X, y)



X_train, X_test, y_train, y_test = s.split_and_scale(X, y)


params={'alpha': l.lognuniform(low=-4,high=2,size=75, base=10),
        'gamma': l.lognuniform(low=-4,high=2,size=75, base=10)
        }

kpca_r_bs=k.BayesKernelPCA(X_train, params, n_iter=200, kernel='rbf')

kpca_r_best_params=kpca_r_bs.get_best_params()
score_r=kpca_r_bs.get_results().iloc[0,-1]

kpca_=KernelPCA(**kpca_r_best_params, kernel='rbf').fit(X_train)
rbf_eigen=np.cumsum(kpca_.lambdas_)/np.sum(kpca_.lambdas_)
eigen_plot_data=pd.DataFrame({'PC':range(len(rbf_eigen)),'eigen':rbf_eigen, 'kernel_type':['rbf']*len(rbf_eigen)})

print('rbf done!')




poly_params={'alpha': l.lognuniform(low=-3,high=2,size=60, base=10),
        'gamma': l.lognuniform(low=-4,high=2,size=60, base=10),
        'degree': [2,3,4,5,6,7,8,9,10],
        'coef0': l.lognuniform(low=-4,high=2,size=60, base=10)
        }

kpca_p_bs=k.BayesKernelPCA(X_train, poly_params, n_iter=200, kernel='poly')

kpca_p_best_params=kpca_p_bs.get_best_params()
score_p=kpca_p_bs.get_results().iloc[0,-1]

kpca_p_best_params.update({'score':kpca_p_bs.get_results().iloc[0,-1]})

kpca_=KernelPCA(**kpca_p_best_params, kernel='poly').fit(X_train)
poly_eigen=np.cumsum(kpca_.lambdas_)/np.sum(kpca_.lambdas_)
aux=pd.DataFrame({'PC':range(len(poly_eigen)),'eigen':poly_eigen, 'kernel_type':['poly']*len(poly_eigen)})
eigen_plot_data=pd.concat([eigen_plot_data, aux], axis=0)

print('poly done!')





if score_r > score_p:
    ker='rbf'
    kpca_params=kpca_r_best_params
elif score_p > score_r:
    ker='poly'
    kpca_params=kpca_p_best_params





kpca_r_best_params.update({'score':kpca_r_bs.get_results().iloc[0,-1]})
tmp1=pd.DataFrame(kpca_r_best_params, index=['rbf'])

kpca_p_best_params.update({'score':kpca_p_bs.get_results().iloc[0,-1]})
series=pd.Series(kpca_p_best_params, name='poly')
tmp1=tmp1.append(series)


tmp1.to_csv(kernel_params_path)





#kpca=KernelPCA(alpha=0.0001, gamma=0.005, kernel='rbf').fit(X_train)
kpca=KernelPCA(**kpca_params, kernel=ker).fit(X_train)

print('kpca done!')







X_train=kpca.transform(X_train)#[:,:110]
X_test=kpca.transform(X_test)#[:,:110]







params={'C': l.lognuniform(low=-3,high=2,size=75, base=10),
        'gamma': l.lognuniform(low=-4,high=2,size=75, base=10)
        }

metrics=make_scorer(roc_auc_score)
clf=SVC(kernel='rbf')
rbf_bs=g.GridSearch(X_train, y_train, clf, metrics, params)

rbf_best_params=rbf_bs.get_best_params()

#clf=SVC(kernel='rbf', C=27.43, gamma=2.09)
clf=SVC(kernel='rbf',**rbf_best_params)
clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)
cm=confusion_matrix(y_test, y_pred)
print(cm)
print(classification_report(y_test, y_pred))
print('ROC:',roc_auc_score(y_test, y_pred))
print('Accuracy:',accuracy_score(y_test, y_pred))
print('f1_score:',f1_score(y_test, y_pred))
print('DOC:', cm[0,0]*cm[1,1]/(cm[1,0]*cm[0,1]))
rbf_best_params.update({'score':rbf_bs.get_results().iloc[0,-1]})

tmp=pd.DataFrame(rbf_best_params, index=['rbf'])

print('rbf done!')



gs=GridSearchCV(estimator=clf,
            scoring=metrics,
            param_grid=params,
            n_jobs=-1
            ).fit(X_train,y_train)

gs_results=pd.DataFrame(gs.cv_results_)



poly_params={'C': l.lognuniform(low=-3,high=2,size=75, base=10),
        'gamma': l.lognuniform(low=-4,high=2,size=75, base=10),
        'degree': [2,3,4,5,6,7,8,9,10],
        'coef0': l.lognuniform(low=-4,high=2,size=75, base=10)
        }
clf=SVC(kernel='poly')
poly_bs=g.BayesSearch(X_train, y_train, clf, metrics, poly_params, n_iter=150)
poly_best_params=poly_bs.get_best_params()
clf=SVC(kernel='poly',**poly_best_params)
clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)
cm=confusion_matrix(y_test, y_pred)
print(cm)
print(classification_report(y_test, y_pred))
print('ROC:',roc_auc_score(y_test, y_pred))
print('Accuracy:',accuracy_score(y_test, y_pred))
print('f1_score:',f1_score(y_test, y_pred))
print('DOC:', cm[0,0]*cm[1,1]/(cm[1,0]*cm[0,1]))
poly_best_params.update({'score':poly_bs.get_results().iloc[0,-1]})
series=pd.Series(poly_best_params, name='poly')

tmp=tmp.append(series)

print('poly done!')






sigmoid_params={'C': l.lognuniform(low=-3,high=2,size=75, base=10),
        'gamma': l.lognuniform(low=-4,high=2,size=75, base=10),
        'coef0': l.lognuniform(low=-4,high=2,size=75, base=10)
        }
clf=SVC(kernel='sigmoid')
sigmoid_bs=g.BayesSearch(X_train, y_train, clf, metrics, poly_params, n_iter=150)
sigmoid_best_params=sigmoid_bs.get_best_params()
clf=SVC(kernel='sigmoid',**sigmoid_best_params)
clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)
cm=confusion_matrix(y_test, y_pred)
print(cm)
print(classification_report(y_test, y_pred))
print('ROC:',roc_auc_score(y_test, y_pred))
print('Accuracy:',accuracy_score(y_test, y_pred))
print('f1_score:',f1_score(y_test, y_pred))
print('DOC:', cm[0,0]*cm[1,1]/(cm[1,0]*cm[0,1]))
sigmoid_best_params.update({'score':sigmoid_bs.get_results().iloc[0,-1]})
series=pd.Series(sigmoid_best_params, name='sigmoid')

tmp=tmp.append(series)

print('sigmoid done!')

tmp.to_csv(model_params_path)


from sklearn.model_selection import LeaveOneOut

from sklearn.preprocessing import MaxAbsScaler
scaling = MaxAbsScaler().fit(X)
X2 = pd.DataFrame(scaling.transform(X), index=X.index, columns=X.columns)
loo = LeaveOneOut()
loo.get_n_splits(X2)


kpca2=KernelPCA(**kpca_r_best_params, kernel='rbf').fit(X2)

scores=[]   
sizes=[25,50,75,100,110,130,175,200,250,300,400,500]

for i in sizes:
    X3=kpca.transform(X2)[:,:i]
    print(i)
    y_pred=[]
    for train_index, test_index in loo.split(X3):
        X_train, X_test = X3[train_index], X3[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf=SVC(kernel='rbf',**rbf_best_params)
        clf.fit(X_train, y_train)
        y_pred.append(clf.predict(X_test))
    scores.append(roc_auc_score(y, y_pred))


cm=confusion_matrix(y, y_pred)
print(cm)
print(classification_report(y, y_pred))
print('ROC:',roc_auc_score(y, y_pred))
print('Accuracy:',accuracy_score(y, y_pred))
print('f1_score:',f1_score(y, y_pred))
print('DOC:', cm[0,0]*cm[1,1]/(cm[1,0]*cm[0,1]))

############################# Kernel FDA ###########################################
param = {'kernel' : ['linear', 'rbf', 'poly', 'sigmoid', 'laplacian', 'chi2'],
         'n_components' : [1],
         'robustness_offset': [1e-9, 1e-8, 1e-7, 1e-6]}
gsKernel = GridKernelFDA(X_train, y_train, param, metrics=['accuracy', 'f1', 'recall'])
best_params = gsKernel.get_best_params()

cls = Kfda(**best_params)
cls.fit(X_train, y_train)
preds = cls.predict(X_test)

reportFDA = classification_report(y_test, preds)
cm = confusion_matrix(y_test, preds)
print(reportFDA)
print(cm)


############################# LDA ###########################################
clsLDA = LinearDiscriminantAnalysis(n_components=1)
clsLDA.fit(X_train, y_train)
predsLDA = clsLDA.predict(X_test)

reportLDA = classification_report(y_test, predsLDA)
cmLDA = confusion_matrix(y_test, predsLDA)
print(reportLDA)
print(cmLDA)


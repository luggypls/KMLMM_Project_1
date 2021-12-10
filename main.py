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



kpca_p_best_params={'alpha':0.00117,
                    'coef0':6.28968,
                    'degree':4,
                    'gamma':0.000176}

kpca_r_best_params={'alpha':0.000281,
                    'gamma':0.006779
                    }

kpca_r_best_params2={'alpha': 0.0012555159579715515,
                     'gamma': 0.01197665920283543
                    }

poly_best_params={'C':4.4414,
                  'coef0':23.416,
                  'degree':10,
                  'gamma':0.2862}

rbf_best_params={'C':50.1047,
                 'gamma':0.04299}

sigmoid_best_params={'C':16.186630258905687,
                  'coef0':0.00016749119590638508,
                  'gamma':0.020910019759823182}

rbf_best_params2={'C':10.413418204791542,
                  'gamma':1.8657583531926265

}




data_path='./Data/pd_speech_features.csv'
kernel_params_path='./Params/Kpca_params1.csv'
model_params_path='./Params/model_params1.csv'

X,y = l.load_data(data_path)


from sklearn.preprocessing import MaxAbsScaler
scaling = MaxAbsScaler().fit(X)
X2 = pd.DataFrame(scaling.transform(X), index=X.index, columns=X.columns)

X_train, X_test, y_train, y_test = s.split_and_scale(X, y)


params={'alpha': l.lognuniform(low=-4,high=2,size=75, base=10),
        'gamma': l.lognuniform(low=-4,high=2,size=75, base=10)
        }

kpca_r_bs=k.BayesKernelPCA(X_train, params, n_iter=300, kernel='rbf')

kpca_r_best_params=kpca_r_bs.get_best_params()
score_r=kpca_r_bs.get_results().iloc[0,-1]

kpca_=KernelPCA(**kpca_r_best_params, kernel='rbf').fit(X2)
rbf_eigen=np.cumsum(kpca_.lambdas_)/np.sum(kpca_.lambdas_)
eigen_plot_data=pd.DataFrame({'PC':range(len(rbf_eigen)),'eigen':rbf_eigen, 'kernel_type':['rbf']*len(rbf_eigen)})

print('rbf done!')




poly_params={'alpha': l.lognuniform(low=-3,high=2,size=75, base=10),
        'gamma': l.lognuniform(low=-4,high=2,size=75, base=10),
        'degree': [2,3,4,5,6,7,8,9,10],
        'coef0': l.lognuniform(low=-4,high=2,size=75, base=10)
        }

kpca_p_bs=k.BayesKernelPCA(X_train, poly_params, n_iter=300, kernel='poly')

kpca_p_best_params=kpca_p_bs.get_best_params()
score_p=kpca_p_bs.get_results().iloc[0,-1]

kpca_p_best_params.update({'score':kpca_p_bs.get_results().iloc[0,-1]})

kpca_=KernelPCA(**kpca_p_best_params, kernel='poly').fit(X2)
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
rbf_bs=g.BayesSearch(X_train, y_train, clf, metrics, params, n_iter=300)

rbf_best_params=rbf_bs.get_best_params()

#clf=SVC(kernel='rbf', C=27.43, gamma=2.09)
clf=SVC(kernel='rbf',**rbf_best_params)
clf.fit(X_train, y_train)
print(confusion_matrix(y_test, clf.predict(X_test)))
print(classification_report(y_test, clf.predict(X_test)))

rbf_best_params.update({'score':rbf_bs.get_results().iloc[0,-1]})

tmp=pd.DataFrame(rbf_best_params, index=['rbf'])

print('rbf done!')





poly_params={'C': l.lognuniform(low=-3,high=2,size=75, base=10),
        'gamma': l.lognuniform(low=-4,high=2,size=75, base=10),
        'degree': [2,3,4,5,6,7,8,9,10],
        'coef0': l.lognuniform(low=-4,high=2,size=75, base=10)
        }
clf=SVC(kernel='poly')
poly_bs=g.BayesSearch(X_train, y_train, clf, metrics, poly_params, n_iter=300)
poly_best_params=poly_bs.get_best_params()
clf=SVC(kernel='poly',**poly_best_params)
clf.fit(X_train, y_train)
print(confusion_matrix(y_test, clf.predict(X_test)))
print(classification_report(y_test, clf.predict(X_test)))
poly_best_params.update({'score':poly_bs.get_results().iloc[0,-1]})
series=pd.Series(poly_best_params, name='poly')

tmp=tmp.append(series)

print('poly done!')






sigmoid_params={'C': l.lognuniform(low=-3,high=2,size=75, base=10),
        'gamma': l.lognuniform(low=-4,high=2,size=75, base=10),
        'coef0': l.lognuniform(low=-4,high=2,size=75, base=10)
        }
clf=SVC(kernel='sigmoid')
sigmoid_bs=g.BayesSearch(X_train, y_train, clf, metrics, poly_params, n_iter=300)
sigmoid_best_params=sigmoid_bs.get_best_params()
clf=SVC(kernel='sigmoid',**sigmoid_best_params)
clf.fit(X_train, y_train)
print(confusion_matrix(y_test, clf.predict(X_test)))
print(classification_report(y_test, clf.predict(X_test)))
sigmoid_best_params.update({'score':sigmoid_bs.get_results().iloc[0,-1]})
series=pd.Series(sigmoid_best_params, name='sigmoid')

tmp=tmp.append(series)




p=sns.lineplot(data=eigen_plot_data,x='PC', y='eigen', hue='kernel_type')
p.set_xlabel("# PC")
p.set_ylabel("% Explained Variance")


# =============================================================================
# 
# [[109  83]
#  [ 12 552]]
# 
# [[106  86]
#  [ 12 552]]   #rbf_best_params all
# 
# [[124  68]
#  [ 14 550]]   #+rbf_best_params X[:300]
# 
# [[129  63]
#  [ 21 543]]   #+rbf_best_params X[:200]
# 
# [[138  54]
#  [ 21 543]]   #rbf_best_params X[:150]
# 
# [[140  52]
#  [ 23 541]]   #rbf_best_params X[:110]
# 
# =============================================================================





tmp.to_csv(model_params_path)
# =============================================================================
# 
# 
# import matplotlib.pyplot as plt
# 
# 
# 
# 
# from sklearn.model_selection import LeaveOneOut
# 
# from sklearn.preprocessing import MaxAbsScaler
# scaling = MaxAbsScaler().fit(X)
# X2 = pd.DataFrame(scaling.transform(X), index=X.index, columns=X.columns)
# loo = LeaveOneOut()
# loo.get_n_splits(X2)
# 
# 
# kpca2=KernelPCA(**kpca_p_best_params, kernel='poly').fit(X2)
# 
# scores=[]   
# sizes=[25,50,75,100,110,130,175,200,250,300,400,500]
# 
# for i in sizes:
#     X3=kpca.transform(X2)[:,:100]
#     print(i)
#     y_pred=[]
#     for train_index, test_index in loo.split(X3):
#         X_train, X_test = X3[train_index], X3[test_index]
#         y_train, y_test = y[train_index], y[test_index]
#         clf=SVC(kernel='rbf',**rbf_best_params2)
#         clf.fit(X_train, y_train)
#         y_pred.append(clf.predict(X_test))
#         
#     scores.append(roc_auc_score(y, y_pred))
# 
# 
# 
# print(confusion_matrix(y, y_pred))
# print(classification_report(y, y_pred))
# print(roc_auc_score(y, y_pred))
# 
# p=sns.lineplot(x=sizes,y=scores)
# p.set_xlabel("# PC")
# p.set_ylabel("ROC score")
# 
# #rbf with p kernel 75 pc for 0.817
# #poly with p kernel 250 pc for 0.811
# #sigmoid 0.775
# #rbf with r kernel from no num csv with 100PC : 0.855
# 
# =============================================================================

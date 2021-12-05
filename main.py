import loading as l
import gridsearch as g

import numpy as np
from sklearn.svm import SVC,LinearSVC
from sklearn.metrics import make_scorer, roc_auc_score, recall_score, accuracy_score,confusion_matrix, precision_score

from sklearn.model_selection import train_test_split


data_path='./Data/pd_speech_features.csv'

X,y = l.load_data(data_path)


# Author: Sheikh Rabiul Islam
# Date: 03/14/2019
# Purpose: Random Forest on fully processed data

#import modules
import pandas as pd   
import numpy as np
import time

#configurations
config_file = 'config.txt'
config = pd.read_csv(config_file,sep=',', index_col =None)
resample_data = config.iloc[0,1] #0 or 1

print("SVM:",resample_data)


start = time.time()

from sklearn.model_selection import KFold, cross_val_score

from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0, probability=True)
#from sklearn import svm
#classifier = svm.SVC(C=1.0, kernel='rbf', degree=3, gamma='auto_deprecated', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', random_state=None)
# import processed data
X_train = np.load('data/data_fully_processed_X_train.npy')
y_train = np.load('data/data_fully_processed_y_train.npy')

if resample_data == 1:
    X_train = np.load('data/data_fully_processed_X_train_resampled.npy')
    y_train = np.load('data/data_fully_processed_y_train_resampled.npy')


X_test = np.load('data/data_fully_processed_X_test.npy')
y_test = np.load('data/data_fully_processed_y_test.npy')

# Fitting classifier to the Training set    
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Performance metrics
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, roc_curve, auc, precision_recall_curve 
from sklearn.metrics import precision_recall_fscore_support
cm = confusion_matrix(y_test, y_pred)

#accuracy -number of instance correctly classified
acsc = accuracy_score(y_test, y_pred) 
df_cm = pd.DataFrame([[cm[1][1], cm[0][0],cm[0][1], cm[1][0]]], 
                        index=[0],
                        columns=['True Positives','True Negatives', 'False Positives', 'False Negatives'])
print(df_cm)
#precision, recall, fscore, support
precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred,average='binary')

#balanced_as = balanced_accuracy_score(y_test, y_pred)

fpr, tpr, thresholds = roc_curve(y_test, classifier.predict_proba(X_test)[:,1], pos_label=1)
roc_auc = auc(fpr,tpr) # ROC-AUC

#precision recall AUC ->PRC
prc_precision, prc_recall, prc_thresholds = precision_recall_curve(y_test, classifier.predict_proba(X_test)[:,1])
#prc_auc = auc(prc_precision,prc_recall)
prc_auc = ''
df_metrics = pd.DataFrame([[acsc, precision, recall, fscore,roc_auc]], 
                        index=[0],
                        columns=['accuracy','precision', 'recall', 'fscore', 'ROC-AUC'])

print(df_metrics)
end = time.time()
print("Time taken:", end-start)
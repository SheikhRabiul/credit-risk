# Author: Sheikh Rabiul Islam
# Date: 03/14/2019
# Purpose: Random Forest on fully processed data

#import modules
import pandas as pd   
import numpy as np
import time

#configurations
resample_data = 0   #set 1 to use resampled training set, 0 to use default (imbalanced) training set

from sklearn.model_selection import KFold, cross_val_score

## random forest
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
classifier = QuadraticDiscriminantAnalysis()

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
prc_auc = auc(prc_precision,prc_recall)

df_metrics = pd.DataFrame([[acsc, precision, recall, fscore,roc_auc,prc_auc]], 
                        index=[0],
                        columns=['accuracy','precision', 'recall', 'fscore', 'ROC-AUC','PRC-AUC'])

print(df_metrics)
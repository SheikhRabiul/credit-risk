# Author: Sheikh Rabiul Islam
# Date: 02/10/2019
# Purpose: 

#import modules
import pandas as pd   
import numpy as np
import time



## apply different classifer from below, uncomment the one you like to be in action
from sklearn.model_selection import KFold, cross_val_score
classifier=''
y_pred_all=np.empty(shape=(0)) #empty 1d numpy array
proba_all=np.empty(shape=(0,2)) # empty 2d numpy array-> o rows 2 column

## SVM
#from sklearn.svm import SVC
#classifier = SVC(kernel = 'linear', random_state = 0, probability=True)

## random forest
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=10,criterion="gini")


#print('Train: %s | test: %s' % (train_indices, test_indices))      
X_train = np.load('data/data_fully_processed_X_train_resampled.npy')
y_train = np.load('data/data_fully_processed_y_train_resampled.npy')

#X_train = np.load('data/result_numeric_one_hot_encoded_X_train.npy')
#y_train = np.load('data/result_numeric_one_hot_encoded_y_train.npy')


X_test = np.load('data/data_fully_processed_X_test.npy')
y_test = np.load('data/data_fully_processed_y_test.npy')

# Fitting SVM to the Training set    
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)


proba = classifier.predict_proba(X_test)



# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix,accuracy_score, precision_score
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

df_metrics = pd.DataFrame([[acsc, precision, recall, fscore]], 
                        index=[0],
                        columns=['accuracy','precision', 'recall', 'fscore'])

print(df_metrics)

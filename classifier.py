# Author: Sheikh Rabiul Islam
# Date: 02/10/2019
# Purpose: 

#import modules
import pandas as pd   
import numpy as np
import time


# import data
dataset = pd.read_csv('data/result.csv', sep=',')


#move default column to the end and delete unnecessary columns.
defaulted = dataset['defaulted']
dataset.drop(labels=['defaulted','currentLoanDelinquencyStatus','Unnamed: 0','loanSequenceNumber','loanSequenceNumber.1','sellerName','servicerName','yr.1','qr.1','preHarpLoanSequenceNumber'], axis=1, inplace = True)
dataset.insert(len(dataset.columns),'defaulted', defaulted)


# seperate the dependent (target) variaable
X = dataset.iloc[:,0:-1].values
X_y =dataset.iloc[:,0:-1]
X_columns = dataset.iloc[:,0:-1].columns.values
y = dataset.iloc[:,-1].values

X_cloumns_d = { X_columns[i]:i for i in range(0, len(X_columns)) }

for i in range(0,len(X_columns)):
    print(i,',',X_columns[i])
    
"""
0 , monthlyReportingPeriod
1 , currentActualUPB
2 , loanAge
3 , remainingMonthToLegalMaturity
4 , repurchaseFlag
5 , modificationFlag
6 , zeroBalanceCode
7 , zeroBalanceEffectiveDate
8 , currentInterestRate
9 , currentDeferredUPB
10 , dueDateOfLastPaidInstallment
11 , miRecoveries
12 , netSalesProceeds
13 , nonMiRecoveries
14 , expenses
15 , legalCosts
16 , maintenanceAndPreservationCosts
17 , taxesAndInsurance
18 , miscellaneousExpenses
19 , actualLossCalculation
20 , modificationCost
21 , stepModificationFlag
22 , deferredPaymentModification
23 , estimatedLoandToValue
24 , yr
25 , qr
26 , creditScore
27 , firstPaymentDate
28 , firstTimeHomeBuyerFlag
29 , maturityDate
30 , metropolitanDivisionOrMSA
31 , mortgageInsurancePercentage
32 , numberOfUnits
33 , occupancyStatus
34 , originalCombinedLoanToValue
35 , originalDebtToIncomeRatio
36 , originalUPB
37 , originalLoanToValue
38 , originalInterestRate
39 , channel
40 , prepaymentPenaltyMortgageFlag
41 , productType
42 , propertyState
43 , propertyType
44 , postalCode
45 , loanPurpose
46 , originalLoanTerm
47 , numberOfBorrowers
48 , superConformingFlag
"""

# Encoding categorical data

nominal_l = ['monthlyReportingPeriod','repurchaseFlag', 'modificationFlag', 'zeroBalanceEffectiveDate', 'dueDateOfLastPaidInstallment','netSalesProceeds','stepModificationFlag','deferredPaymentModification', 'yr', 'qr','firstPaymentDate','firstTimeHomeBuyerFlag','maturityDate','metropolitanDivisionOrMSA', 'occupancyStatus', 'channel','prepaymentPenaltyMortgageFlag','productType','propertyState','propertyType','postalCode','loanPurpose','superConformingFlag']
ordinal_l = [ ] # we don't have any feature that requires to preserve order after encoding.

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()

#print(X[:, 5])

nominal_indexes = []
for j in range(0,len(nominal_l)):
    i = X_cloumns_d[nominal_l[j]]
    nominal_indexes.append(i)
    print("executing ",nominal_l[j], " i:",i)
    X[:, i] = labelencoder_X.fit_transform(X[:, i])
    
#print(X)
onehotencoder = OneHotEncoder(categorical_features = nominal_indexes)
X = onehotencoder.fit_transform(X).toarray()

#print(X)


# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Feature Scaling (scaling all attributes/featues in the same scale)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

# excluding few features those has none or little effect on classification result.
# This will simplyfy the model (less overfitting)
#X= X[:, :-2]


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

##    Naive Bayes
#from sklearn.naive_bayes import BernoulliNB
#classifier = BernoulliNB() 

# gradient Boosting
#from sklearn.ensemble import GradientBoostingClassifier
#classifier = GradientBoostingClassifier()

# knn
#from sklearn.neighbors import KNeighborsClassifier
#classifier = KNeighborsClassifier()

# Extra Trees 
#from sklearn.ensemble import ExtraTreesClassifier
#classifier = ExtraTreesClassifier(criterion="entropy")

##we found extra Trees classifier was the best classifier with highest accuracy.To see other classifiers result 
#keep that two line of code uncommented 


# k fold cross validation

k_fold = KFold(n_splits=10)
start = time.time()
for train_indices, test_indices in k_fold.split(X):
    #print('Train: %s | test: %s' % (train_indices, test_indices))      
    X_train = X[train_indices[0]:train_indices[-1]+1]
    y_train = y[train_indices[0]:train_indices[-1]+1]

    X_test = X[test_indices[0]:test_indices[-1]+1]
    y_test = y[test_indices[0]:test_indices[-1]+1]

    # Fitting SVM to the Training set    
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    y_pred_all=np.concatenate((y_pred_all,y_pred),axis=0)

    proba = classifier.predict_proba(X_test)
    proba_all=np.concatenate((proba_all,proba))
    
end = time.time()
diff = end - start
print("classification time:")
print(diff)
# this gives us how strong is the ye/no decision with a probability value (continuous value 
# rather than just the discrete binary decision)
df_result = pd.DataFrame(proba_all,columns=['probability_no','probability_yes'])


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix,accuracy_score, precision_score
from sklearn.metrics import precision_recall_fscore_support
cm = confusion_matrix(y, y_pred_all)

#accuracy -number of instance correctly classified
acsc = accuracy_score(y, y_pred_all) 
df_cm = pd.DataFrame([[cm[1][1], cm[0][0],cm[0][1], cm[1][0]]], 
                        index=[0],
                        columns=['True Positives','True Negatives', 'False Positives', 'False Negatives'])
print(df_cm)
#precision, recall, fscore, support
precision, recall, fscore, support = precision_recall_fscore_support(y, y_pred_all,average='binary')

df_metrics = pd.DataFrame([[acsc, precision, recall, fscore]], 
                        index=[0],
                        columns=['accuracy','precision', 'recall', 'fscore'])

print(df_metrics)

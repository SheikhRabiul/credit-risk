# Author: Sheikh Rabiul Islam
# Date: 02/10/2019; updated: 3/14/2019
# Purpose: remove unimportant features; labelEncode; onHotEncode;Scale; resample minority class;
#		save the fully processed data as numpy array (binary: data/____.npy)

#import modules
import pandas as pd   
import numpy as np


# import data
dataset = pd.read_csv('data/data_preprocessed.csv', sep=',')


#move default column to the end and delete unimportnat columns (selected by features selection approaches (filter(RF) and wrapper(correlation)) before).
defaulted = dataset['defaulted']
dataset.drop(labels=['defaulted','sellerName','servicerName','Unnamed: 0','loanSequenceNumber.1','yr','qr','yr.1','qr.1','zeroBalanceCode','zeroBalanceEffectiveDate','loanSequenceNumber','preHarpLoanSequenceNumber','taxesAndInsurance','superConformingFlag',\
                     'repurchaseFlag','remainingMonthToLegalMaturity','originalLoanTerm','numberOfBorrowers','monthlyReportingPeriod','modificationCost','miscellaneousExpenses',\
                     'miRecoveries','maintenanceAndPreservationCosts','loanAge','estimatedLoandToValue','firstPaymentDate','maturityDate','monthlyReportingPeriod','dueDateOfLastPaidInstallment'], axis=1, inplace = True)
dataset.insert(len(dataset.columns),'defaulted', defaulted)


# seperate the dependent (target) variaable
X = dataset.iloc[:,0:-1].values
X_columns = dataset.iloc[:,0:-1].columns.values
y = dataset.iloc[:,-1].values

X_cloumns_d = { X_columns[i]:i for i in range(0, len(X_columns)) }
for i in range(0,len(X_columns)):
    print(i,',',X_columns[i])

# Encoding categorical data
nominal_l = ['modificationFlag','netSalesProceeds','stepModificationFlag','deferredPaymentModification','firstTimeHomeBuyerFlag','metropolitanDivisionOrMSA', 'occupancyStatus', 'channel','prepaymentPenaltyMortgageFlag','productType','propertyState','propertyType','postalCode','loanPurpose']
ordinal_l = [ ] # we don't have any feature that requires to preserve order after encoding.

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()

nominal_indexes = []
for j in range(0,len(nominal_l)):
    i = X_cloumns_d[nominal_l[j]]
    nominal_indexes.append(i)
    #print("executing ",nominal_l[j], " i:",i)
    X[:, i] = labelencoder_X.fit_transform(X[:, i])
    
df_dump_part1 = pd.DataFrame(X, columns=X_columns)
df_dump_part2 = pd.DataFrame(y, columns=['defaulted'])   
df_dump = pd.concat([df_dump_part1,df_dump_part2], axis = 1)     
df_dump.to_csv("data/data_preprocessed_numerical.csv",encoding='utf-8')   # keeping a backup of preprocessed numerical data.

onehotencoder = OneHotEncoder(categorical_features = nominal_indexes)
X = onehotencoder.fit_transform(X).toarray()

# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Feature Scaling (scaling all attributes/featues in the same scale)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)



#########seperating training and test set  ##################
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=42, stratify=y)
del(X)   # free some memory; encoded (onehot) data takes lot of memory
del(y) # free some memory; encoded (onehot) data takes lot of memory

#dump onehot encoded training data
# save the fully processed data as binary for future use in any ML algorithm without any more preprocessing. 
np.save('data/data_fully_processed_X_train.npy',X_train)
np.save('data/data_fully_processed_y_train.npy',y_train)

print("Before OverSampling, counts of label '1': {}".format(sum(y_train==1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train==0)))

# save the fully processed data as binary for future use in any ML algorithm without any more preprocessing. 
np.save('data/data_fully_processed_X_test.npy',X_test)
np.save('data/data_fully_processed_y_test.npy',y_test)


################oversampling the minority class of training set #########

from imblearn.over_sampling import SMOTE 
# help available here: #https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.SMOTE.html
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train)

# save the fully processed data as binary for future use in any ML algorithm without any more preprocessing. 
np.save('data/data_fully_processed_X_train_resampled.npy',X_train_res)
np.save('data/data_fully_processed_y_train_resampled.npy',y_train_res)

print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))
print("After OverSampling, counts of label '1': {}".format(sum(y_train_res==1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_train_res==0)))


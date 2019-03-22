# Author: Sheikh Rabiul Islam
# Date: 3/20/2019
# Purpose: run following algorithms; classifiers.ga might not run on windows, it runs on linux machine
#	classifier_lr.py -> Logistic Regression (LR)
#	classifier_dt.py -> Decision Tree (DT)
#	classifier_rf.py -> Random Forest (RF)
#	classifier_et -> Extra Trees (ET)
#	classifier_gradient_boosting.py - > Gradient Boosting (GB)
#	classifier_adaboost.py -> Adaboost
#	classifier_nb.py -> Naive Bayes
#	clasifier_mda.py -> Multiple Discriminant Analysis (MDA)
#	classifier_svm.py -> Support Vector Machine (SVM)
#	classifier_rough_set.py -> Rough Set (RS)
#	classifier_ann.py -> Artificial Neural Network (ANN)
#	classifier_ga.py -> Genetic Algorithm (GA)

import time
s = time.time()

start = time.time()
exec(open("classifier_lr.py").read())
end = time.time()
print("\n\nTime taken by classifier_lr.py:", end-start)

start = time.time()
exec(open("classifier_dt.py").read())
end = time.time()
print("\n\nTime taken by classifier_dt.py:", end-start)

start = time.time()
exec(open("classifier_rf.py").read())
end = time.time()
print("\n\nTime taken by classifier_rf.py:", end-start)

start = time.time()
exec(open("classifier_et.py").read())
end = time.time()
print("\n\nTime taken by classifier_et.py:", end-start)

start = time.time()
exec(open("classifier_gradient_boosting.py").read())
end = time.time()
print("\n\nTime taken by classifier_gradient_boosting.py:", end-start)

start = time.time()
exec(open("classifier_adaboost.py").read())
end = time.time()
print("\n\nTime taken by classifier_adaboost.py:", end-start)

start = time.time()
exec(open("classifier_nb.py").read())
end = time.time()
print("\n\nTime taken by classifier_nb.py:", end-start)

start = time.time()
exec(open("classifier_mda_qda.py").read())
end = time.time()
print("\n\nTime taken by classifier_mda_qda.py:", end-start)

start = time.time()
exec(open("classifier_svm.py").read())
end = time.time()
print("\n\nTime taken by classifier_svm.py:", end-start)

start = time.time()
exec(open("classifier_roughsets.py").read())
end = time.time()
print("\n\nTime taken by classifier_roughsets.py:", end-start)

start = time.time()
exec(open("classifier_ann.py").read())
end = time.time()
print("\n\nTime taken by classifier_ann.py:", end-start)

e = time.time()
print("\n\nTotal Time taken by all classifiers.py:", e-s)



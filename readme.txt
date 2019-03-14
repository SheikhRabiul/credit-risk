process_data.py -> preprocess data by running following 4 files
	data_extract.py -> read the dataset; store it in sqlite database.
	data_transform.py -> clean data; sample data
	data_load.py -> move cleaned and sampled data from raw tables to main table. Also write into data/result.csv.
	data_conversion -> remove unimportant features; labelEncode; onHotEncode;Scale; resample minority class;
		save the fully processed data as numpy array (binary: data/____.npy)

run_classifiers.py
	classifier_lr.py -> Logistic Regression (LR)
	classifier_dt.py -> Decision Tree (DT)
		classifier_rf.py -> Random Forest (RF)
		classifier_et -> Extra Trees (ET)
	clasifier_mda.py -> Multiple Discriminant Analysis (MDA)
	classifier_svm.py -> Support Vector Machine (SVM)
	classifier_rough_set.py -> Rough Set (RS)
	classifier_cbr.py -> Case Based Reasoning (CBR)
	classifier_ann.py -> Artificial Neural Network (ANN)
	classifier_ga.py -> Genetic Algorithm (GA)
	  


# Author: Sheikh Rabiul Islam
# Date: 3/14/2019
# Purpose: preprocess data by running following 4 files
#	data_extract.py -> read the dataset; store it in sqlite database.
#	data_transform.py -> clean data; sample data
#	data_load.py -> move cleaned and sampled data from raw tables to main table. Also write into data/data_preprocessed.csv.
#	data_conversion -> remove unimportant features; labelEncode; onHotEncode;Scale; resample minority class;
#		save the fully processed data as numpy array (binary: data/____.npy)

exec(open("data_extract.py").read())

exec(open("data_transform.py").read())

exec(open("data_load.py").read())

exec(open("data_conversion.py").read())

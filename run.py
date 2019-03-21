# Author: Sheikh Rabiul Islam
# Date: 3/20/2019
import sys
sys.stdout = open('print.txt', 'w')  #comment this line in case you want to see output on the console.

import time
s = time.time()
# one set of run is for original data
# another set of run is using resampled data

start = time.time()
exec(open("process_data.py").read())
end = time.time()
print("Time taken by process_data.py:", end-start)

start = time.time()
exec(open("run_classifiers.py").read())
end = time.time()
print("Time taken by run_classifiers.py:", end-start)


import pandas as pd
#change use_resample =1
#configurations
config_file = 'config.txt'
config = pd.read_csv(config_file,sep=',', index_col =None)
resample_data = config.iloc[0,1] 

if resample_data == 1:
    resample_data = 0
else:
    resample_data = 1

config.iloc[0,1] = resample_data
config.to_csv(config_file,encoding='utf-8',index=False)


# another set of run 
start = time.time()
exec(open("run_classifiers.py").read())
end = time.time()
print("Time taken by run_classifiers.py:", end-start)


e = time.time()
print("Total Time taken by all classifiers.py:", e-s)



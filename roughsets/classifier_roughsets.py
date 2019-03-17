#!/usr/bin/env python2.7
# encoding: utf-8
#AUthor:@author Michael Tompkins
#Modified by: Sheikh Rabiul Islam
#Date: 3/17/2019

"""
Some unit tests and usage examples for rough set clustering class
@data UCI Statlog Data Set:
Lichman, M. (2013). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA:
University of California, School of Information and Computer Science.
@author Michael Tompkins
@copyright 2016
"""

# Externals
import time
import json
from collections import Counter
import matplotlib.pyplot as plt
import numpy as npy
from scipy.cluster.vq import kmeans2
import pandas as pd
# Package level imports from /code
from rough_kmeans import *

# Set some rough clustering parameters
maxD = 20			# if None, maxD will be determined by algorithm
max_clusters = 2    # Number of clusters to return

# Set some rough clustering parameters
maxD = 20			# if None, maxD will be determined by algorithm
max_clusters = 2    # Number of clusters to return

# Load some data

df = pd.read_csv("result_numeric_train.csv",sep=',', index_col =None)
response = df['defaulted'].tolist()
df.drop(labels=['Unnamed: 0','defaulted'], inplace = True, axis=1)
header = (df.columns.values).tolist()



#file2 = open("german_all.json","r")
#data = json.load(file2)
#print((list(data.keys())))

#data['response'] = response




# Do some numerical encoding for input payload
#header = []
#data2 = {}
#for key in list(data["payload"].keys()):
#    header.append(key)
#    try:
#        data2[key] = [int(data["payload"][key][m]) for m in range(0,len(data["payload"][key]))]
#        if key == "amount":
#            data2[key] = []
#            for n in range(len(data["payload"][key])):
#                bins = [0,1500,3000,8000,20000]
#                for i,val in enumerate(bins[0:-1]):
#                    if (int(data["payload"][key][n])) >= val and (int(data["payload"][key][n]) < bins[i+1]):
#                        data2[key].append(i+1)
#    except:
#        data2[key] = []
#        encoding = {key : m for m,key in enumerate(Counter(data["payload"][key]).keys())}
#        for n in range(len(data["payload"][key])):
#            data2[key].append(encoding[data["payload"][key][n]])

data2 = {}
for i in range(0, len(header)):
    key = header[i]
    data2[key] = df[key].tolist()
    

clstrk = RoughKMeans(data2,2,0.75,0.25,1.2)
clstrk.get_rough_clusters()
t3 = time.time()
print(("Rough Kmeans Clustering Took: ",t3-t2," secs"))
print(("rough kmeans",clstrk.centroids))

list1 = [i for i in range(len(response)) if response[i] == 0]
list2 = [i for i in range(len(response)) if response[i] == 1]


print(("total instances",Counter(response)))

print(("Rough kmeans groups",len(clstrk.clusters['0']["upper"]),len(clstrk.clusters['1']["upper"])))
print(("Cluster 0 vs Target 0",len(set(clstrk.clusters['0']["upper"]).intersection(set(list1)))))
print(("Cluster 1 vs Target 1",len(set(clstrk.clusters['1']["upper"]).intersection(set(list2)))))
print(("Cluster 0 vs Target 1",len(set(clstrk.clusters['0']["upper"]).intersection(set(list2)))))
print(("Cluster 1 vs Target 0",len(set(clstrk.clusters['1']["upper"]).intersection(set(list1)))))


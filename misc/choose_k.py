#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import math
from sklearn.metrics import classification_report

# Value of k
k = 10
#prefix = 'datafull'

distmatrix = pd.read_csv('WassersteinDistMatrix0.csv',index_col=0)

print(distmatrix.head())
#print(distmatrix.describe())
print(distmatrix.shape)
#(297, 297)
totaln = distmatrix.shape[0]

adjmat = distmatrix.values #convert to numpy array

#print(adjmat)

##make adjmat symmetric

for j in range(0,totaln):

    for i in range(0,j):

        adjmat[i][j] = adjmat[j][i]

##now adjmat is symmetric matrix

#print(adjmat)



# Train: 60%
trainnum = 179
validnum = 59
testnum = 59
#math.floor(0.8*totaln)
print(trainnum)
# 800
trainlist = [i for i in range(0,trainnum)]

# Validation: 20%


validlist = [i for i in range(trainnum,trainnum+validnum)]

df = pd.read_csv('processed.cleveland.dataclean_standardized.csv')
y_true = df['result'].iloc[validlist]
y_true = y_true.values.tolist()


### k-nn algorithm
knn = [[] for i in range(validnum)]

for i in range(0,validnum):
    index = validlist[i]
    # We only use training data for k-nn prediction
    coltobesort = adjmat[0:trainnum,index]
    indexsorted = np.argsort(coltobesort)

    indexsorted = indexsorted[indexsorted != index] #remove validlist[i]/testlist[i] as its own nearest neighbor

    #print(len(indexsorted))

    #len(indexsorted)=684



    knn[i] = indexsorted[0:k]

#print(len(knn))
# len(knn)=59

# Observe nearest neighbors of 1st validation entry
#print('Observe nearest neighbors of validation entry')
#print(knn[0])

    
###


### y_pred
y_pred=[]
for i in range(0,len(knn)):
    knnclass = df['result'].iloc[knn[i]]
    #print(knnclass)
    # sum over the column axis.
    totalsum = knnclass.sum()
    average = totalsum/k
    if (average<0.5):
        y_pred.append(0)
    else:
        y_pred.append(1)

###
print('y_pred:')
print(y_pred)

###code to output to csv
#tempdf = pd.DataFrame(data={"y_pred": y_pred})
#tempdf.to_csv("./y_pred.csv", sep=',',index=False)

# print(len(y_pred))
# 171
print('y_true:')
print(y_true)

target_names =['class 0', 'class 1']
print('Validation results:')
print(classification_report(y_true, y_pred, labels=[0,1],
                            target_names=target_names,
                            digits=4))


### Test Set
testlist = [i for i in range(trainnum+validnum,trainnum+validnum+testnum)]

#df = pd.read_csv('processed.cleveland.dataclean_standardized.csv')
y_true = df['result'].iloc[testlist]
y_true = y_true.values.tolist()


### k-nn algorithm
knn = [[] for i in range(testnum)]

for i in range(0,testnum):
    index = testlist[i]
    # We only use training data for k-nn prediction
    coltobesort = adjmat[0:trainnum,index]
    indexsorted = np.argsort(coltobesort)

    indexsorted = indexsorted[indexsorted != index] #remove validlist[i]/testlist[i] as its own nearest neighbor

    #print(len(indexsorted))

    #len(indexsorted)=684



    knn[i] = indexsorted[0:k]

print(len(knn))
# len(knn)=59

# Observe nearest neighbors of 1st test entry (index 179+59=238)
#print('Observe nearest neighbors of test entry')
#print(knn[0])


    
###


### y_pred
y_pred=[]
for i in range(0,len(knn)):
    knnclass = df['result'].iloc[knn[i]]
    #print(knnclass)
    # sum over the column axis.
    totalsum = knnclass.sum()
    average = totalsum/k
    if (average<0.5):
        y_pred.append(0)
    else:
        y_pred.append(1)

###
print('y_pred:')
print(y_pred)
# print(len(y_pred))
# 171

###code to output to csv
#tempdf = pd.DataFrame(data={"y_pred": y_pred})
#tempdf.to_csv("./y_pred.csv", sep=',',index=False)


print('y_true:')
print(y_true)

target_names =['class 0', 'class 1']
print('Test results:')
print(classification_report(y_true, y_pred, labels=[0,1],
                            target_names=target_names,
                            digits=4))
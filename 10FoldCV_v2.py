#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import math
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

# Value of k for k-NN
k = 16

#best k=16, Overall 10-fold accuracy: 0.8251724137931035


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


#data is already shuffled in data cleaning step
       
#The first n_samples % n_splits folds have size n_samples // n_splits + 1
#, other folds have size n_samples // n_splits, where n_samples is the number of samples.        
         
        
foldsize=math.floor(totaln/10)
bigfoldnum = totaln % 10
#print(bigfoldnum)
#7

#print(foldsize)
#29

fold=[[0]*foldsize]*10

fold[0] = [j for j in range(0,foldsize+1)]
for i in range(1,10):
    if i<bigfoldnum:
        fold[i] = [j for j in range(fold[i-1][-1]+1,fold[i-1][-1]+foldsize+2)]
    else:
        fold[i] = [j for j in range(fold[i-1][-1]+1,fold[i-1][-1]+foldsize+1)]

#for i in range(0,10):
    #print(i)
    #print(fold[i])        


totallist = [i for i in range(0,totaln)]

overallaccsum = 0
overallsensitivitysum = 0
overallspecificitysum = 0
overallprec0sum = 0
overallprec1sum = 0
overallf1_0sum = 0
overallf1_1sum = 0

for foldnum in range(0,10):

    trainlist = set(totallist)-set(fold[foldnum])
    trainlist=list(trainlist)
    #print(trainlist)
    
    ### Test Set
    testlist = fold[foldnum]
    
    df = pd.read_csv('processed.cleveland.dataclean_standardized.csv')
    y_true = df['result'].iloc[testlist]
    y_true = y_true.values.tolist()
    
    
    ### k-nn algorithm
    knn = [[] for i in range(0,len(testlist))]
    
    for i in range(0,len(testlist)):
        # We only use training data for k-nn prediction
        coltobesort = adjmat[trainlist,testlist[i]]
        indexsorted = np.argsort(coltobesort)
        #print(indexsorted)
        
    
        #print(len(indexsorted))
    
        #len(indexsorted)=268
    
    
    
        knn[i] = [trainlist[j] for j in indexsorted[0:k]]
    
    #print(len(knn))
    # len(knn)=29
    
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
    print('Fold '+str(foldnum)+' Test results:')
    print(classification_report(y_true, y_pred, labels=[0,1],
                                target_names=target_names,
                                digits=4))
    
    foldacc = accuracy_score(y_true, y_pred)
    overallaccsum = overallaccsum+foldacc
    
    foldsensitivity = recall_score(y_true, y_pred, pos_label=1)
    overallsensitivitysum = overallsensitivitysum+foldsensitivity
    
    foldspecificity = recall_score(y_true, y_pred, pos_label=0)
    overallspecificitysum = overallspecificitysum+foldspecificity
    
    foldprec0 = precision_score(y_true, y_pred, pos_label=0)
    overallprec0sum = overallprec0sum+foldprec0
    
    foldprec1 = precision_score(y_true, y_pred, pos_label=1)
    overallprec1sum = overallprec1sum+foldprec1
    
    foldf1_0 = f1_score(y_true, y_pred, pos_label=0)
    overallf1_0sum = overallf1_0sum+foldf1_0
    
    foldf1_1 = f1_score(y_true, y_pred, pos_label=1)
    overallf1_1sum = overallf1_1sum+foldf1_1
    #end for loop 10-

overallacc = overallaccsum/10
print('Overall 10-fold accuracy: '+str(overallacc))

overallsensitivity = overallsensitivitysum/10
print('Overall 10-fold sensitivity: '+str(overallsensitivity))

overallspecificity = overallspecificitysum/10
print('Overall 10-fold specificity: '+str(overallspecificity))

overallprec0 = overallprec0sum/10
print('Overall 10-fold precision (class 0): '+str(overallprec0))

overallprec1 = overallprec1sum/10
print('Overall 10-fold precision (class 1): '+str(overallprec1))

overallf1_0 = overallf1_0sum/10
print('Overall 10-fold F1 (class 0): '+str(overallf1_0))

overallf1_1 = overallf1_1sum/10
print('Overall 10-fold F1 (class 1): '+str(overallf1_1))
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Set working directory ###
###

import pandas as pd
pd.set_option('display.max_columns', None)  
import numpy as np
import math
#from sklearn.preprocessing import StandardScaler
import os.path

prefix = 'processed.cleveland.data'

df = pd.read_csv(prefix+'clean_standardized.csv',index_col=0)

print(df.shape)
# (297, 26)
# 25 predictor variabes (index 0 to 24) + 1 result variable
totalrows = df.shape[0]
totalpred = df.shape[1]-1 #25

print(df.head())
print(df.describe())

#create pointclouds folder manually

#create pointcloud files
for i in range(0,totalrows):
    # print(i)
    irow = df.iloc[[i]]
    # drop last column
    irow = irow.iloc[:, :-1]
    pointcloud = irow.copy()
    
    for j in range(0,totalpred):
        newrow = irow.copy()
        newrow.iat[0,j] = 0
        pointcloud = pointcloud.append(newrow,ignore_index=True)
        
    
    pointcloud.to_csv(os.path.join('pointclouds','pc'+str(i)+'.csv'),
                      header=False,index=False)
    
        


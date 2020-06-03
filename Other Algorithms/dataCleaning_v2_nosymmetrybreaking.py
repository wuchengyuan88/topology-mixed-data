#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
pd.set_option('display.max_columns', None)  
import numpy as np
from sklearn.preprocessing import StandardScaler


prefix ='processed.cleveland.data'

df = pd.read_csv(prefix+'.csv',index_col=False,header=None)


print(df.shape)
#(303, 14)

#Drop rows with missing
df = df.replace("?", np.nan)
df = df.dropna()
print(df.shape)
#(297, 14)

#rename columns
df.columns = ['age', 'sex','cp','trestbps','chol',
              'fbs','restecg','thalach','exang',
              'oldpeak','slope','ca','thal','num']

print(df.head())

print(df.describe())

numerical_cols = ['age','trestbps','chol','thalach',
                  'oldpeak','ca']
print(numerical_cols)

#df[numerical_cols] = StandardScaler().fit_transform(df[numerical_cols])

#add new columns (for categorical variables)
#one-hot encoding
categorical_cols = ['sex','cp','fbs','restecg','exang','slope',
                    'thal']

for i in categorical_cols:
    df[i] = pd.to_numeric(df[i], downcast='integer')
    df = pd.concat([df,pd.get_dummies(df[i],prefix=i)],axis=1)
    df = df.drop(columns=i)


#Change final result column to binary
df['result'] = np.where(df['num']==0, 0,1)
df = df.drop(columns='num')
    
#Standardize all columns (except result)
allpredictors = df.columns.tolist()
allpredictors.remove('result')
df[allpredictors] = StandardScaler().fit_transform(df[allpredictors])

print(df.head())

print(df.describe())

print(df.shape)
#(297, 26) --> 25 features, 1 result column

#Symmetry breaking
#for i in range(0,25):
    #df[df.columns[i]]+=(i+5)

print(df.head())
print(df.describe())
print(df.columns)

# shuffle dataframe to faciliate randomized train/validation/test split later
df = df.sample(frac=1,random_state=1)


df.to_csv(prefix+'clean_standardized_nosymmetrybreaking.csv')
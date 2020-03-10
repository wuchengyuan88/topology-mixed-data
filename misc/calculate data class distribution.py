#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import math
from sklearn.metrics import classification_report

df = pd.read_csv('processed.cleveland.dataclean_standardized.csv')

df1 = df['result'].iloc[0:179]

print(df1.describe())

df2 = df['result'].iloc[179:179+59]

print(df2.describe())

df3 = df['result'].iloc[179+59:179+59+59]

print(df3.describe())
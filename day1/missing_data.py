# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("FIRSTDATA.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer = imputer.fit(x[:, 1:])
x[:, 1:] = imputer.transform(x[:, 1:])

print(x)

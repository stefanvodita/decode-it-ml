# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 15:10:59 2019

@author: stefa
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk
#from nltk.stem import PorterStemmer
   
#ps = PorterStemmer() 

useless_words = nltk.download('stopwords')

def simplify(text):
    text = text.lower()
    text = re.sub('[.,:;?!-\']', '', text)
    #text = [word for word in ps.stem(text.split(' '))]
    text = [word for word in text.split(' ') if word not in useless_words]
    print(text)
    return text

# Importing the dataset
dataset = pd.read_csv('decode.tsv', delimiter='\t')
X = dataset.iloc[1:, 0].values
Y = dataset.iloc[1:, 1].values

print(X)
print(Y)

for i, line in enumerate(X):
    X[i] = simplify(line)
    
print(X)

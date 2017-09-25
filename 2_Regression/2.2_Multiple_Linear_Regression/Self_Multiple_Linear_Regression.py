#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 14:01:12 2017

@author: ddxy
"""

#MUltiple Linear Regression

"""
Step 1: importing the libraries
"""

import numpy as np
import matplotlib.pyplot as plt

#panda's library is the best library to import and manage datasets 
import pandas as pd

"""
Step 2: importing the dataset
"""
#2.1: set a working directory by saving this file. The folder that holds this file
#will the be working directory

dataset = pd.read_csv('50_Startups.csv', header = 0)

#Replace some special characters in the header by '_' so that these headers
#can be read correctly in step 6 below
dataset.columns = [x.strip().replace(' ', '_') for x in dataset.columns]
dataset.columns = [x.strip().replace('&', '_') for x in dataset.columns]

#Print the headers
print(list(dataset))

#2.2: create a matrix of features/IVs
X = dataset.iloc[:,:-1].values #[:,:-1] means select all the rows and the 
                               #columns except the last column

#2.3: create a dependent variable vector
y = dataset.iloc[:,4].values

"""
Step 3: Convert categorical data to dummy variables
"""
#first convert the strings in the categorical column to integer coded column
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])

#Second convert the integer coded column into dummy variables
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#Avoiding the Dummy Variable Trap by elimilating the first column because we
#always want to omit 1 dummy variable to fulfill the "independence of errors"
#assumption

X = X[:, 1:]

"""
Step 4: Splitting the dataset into the training and test sets
"""
from sklearn.cross_validation import train_test_split
#random_state = 0 is to make everytime it gives the same result. it is like
#specifying a seed for random() in java
#practically, we do not need this so that every time the program will
#randomly select 20% of data for the test.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


"""
Step 5: Create a Multi_linear_regression machine and let the machine learn
        In other words, create a model
"""
from sklearn.linear_model import LinearRegression
#create the MACHINE
regressor = LinearRegression()
#let the machine LEARN the data
regressor.fit(X_train, y_train)



"""
Step 6: Show statistical results of the model built on the training set
"""
from statsmodels.formula.api import ols

#split the dataset to training and test sets
train_set, test_set = train_test_split(dataset, test_size = 0.2, random_state = 0)

#create the formula string
tempHeaders = list(dataset)
formula = tempHeaders[-1] + " ~ "
for iv in range(0, len(tempHeaders) - 1):
    formula += tempHeaders[iv] + " + "

formula = formula[:-3]

#use ols to create the model
model = ols(formula, train_set).fit()
model.summary()


"""
Step 7: Predicting the Test set results
"""
y_pred = regressor.predict(X_test)



"""
Step 8: Building the optimal model using Backward Elimination
"""

















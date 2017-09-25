#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 12:15:11 2017

@author: ddxy
"""

#SVR (support vector regression)
# Regression Template

"""Step 1: Importing the libraries"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""Step 2: Importing the dataset"""
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

"""Step 3: Splitting the dataset into the Training set and Test set"""
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""


"""Step 4: Feature Scaling"""
#for SVR, we do need to do feature scaling/standardization by ourselves
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()

X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)


"""
Step 5: Fitting SVR Model to the dataset
# Create your regressor here
"""
from sklearn.svm import SVR

#In machine learning, the (Gaussian) radial basis function kernel, or RBF 
#kernel, is a popular kernel function used in various kernelized learning 
#algorithms. In particular, it is commonly used in support vector machine 
#classification
#default kernel of SVR is 'rbf'
regressor = SVR(kernel = 'rbf')
regressor.fit(X, y)


"""Step 6: Predicting a new result"""
#to predict a single value, 
#First, we need to standardize this single value using sc_X
#Second, we need to put the standardized value into a form of matrix
#NOTE that double [] around 6.5 ([[6.5]]) makes 6.5 in a matrix, a single [] 
#around 6.5 makes it a vestor which is not a right type parameter
#The y_pred output is the standardized score, NOT the original value
y_pred = regressor.predict(sc_X.transform(np.array([[6.5]])))

#We now need to inverse transform the y_pred socre to the original value
y_pred = sc_y.inverse_transform(y_pred)



"""Step 7: Visualising the SVR results"""
#In this case, the last tuple (CEO data tuple) is considered an outlier
#this is why the line looks not predicting the last point correctly
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()





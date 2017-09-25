#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 09:47:18 2017

@author: ddxy
"""

#Polynomial Regression

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
#2.1: set a working directory by saving this file to the desired working
#directory. The folder that holds this file will the be working directory
#header = 0 means reads data WITH the header
dataset = pd.read_csv('/home/ddxy/Projects/UdemyAtoZML/2_Regression/2.3_Polynomial_Regression/Position_Salaries.csv', header = 0)

#Replace some special characters in the header by '_' so that these headers
#can be read correctly in step 6 below
#dataset.columns = [x.strip().replace(' ', '_') for x in dataset.columns]
#dataset.columns = [x.strip().replace('&', '_') for x in dataset.columns]
 
#Print the headers
#print(list(dataset))

#2.2: create a matrix of features/IVs
X = dataset.iloc[:,1:-1].values #[:,:-1] means select all the rows and the 
                               #columns except the last column

#2.3: create a dependent variable vector
y = dataset.iloc[:,2].values


"""
NO Step 3 in this case because the sample size is too small

Step 3: Splitting the dataset into the training and test sets

from sklearn.cross_validation import train_test_split
#random_state = 0 is to make everytime it gives the same result. it is like
#specifying a seed for random() in java
#practically, we do not need this so that every time the program will
#randomly select 20% of data for the test.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
"""

"""
Step 3: Fitting Linear Regression to the dataset
"""
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)


"""
Step 4: Fitting Polynomial Regression to the dataset
"""
#use PolynomialFeatures class to transform the original X matrix to a matrix
#containing polynomial terms and assign this transformed matrix to a new 
#matrix used for further LinearRegression
from sklearn.preprocessing import PolynomialFeatures

#the max degree of the power in the model can be easily adjusted by changing
#"degree = 4" below in PolynomialFeatures()
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)

#Use the transformed X matrix -- X_poly and linear regression to create the model
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)


"""
Step 5: Show statistical results of the model built on the training set
"""
from statsmodels.formula.api import ols

#use ols to create the model
model = ols('Salary ~ np.power(Level, 4) + np.power(Level, 3) + np.power(Level, 2) + Level', dataset).fit()
print(model.summary())

"""
Step 6: Visualising the Linear and polynomial Regression results
"""
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
#use poly_reg.fit_transform(X) so that lin_reg_2 can be used for predicting
#any new matrix of X
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'green')
plt.title('Truth or Bluff')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

"""
To smooth the line, we can improve above plotting method with the one below
"""
X_smooth = np.arange(min(X), max(X), 0.1)
X_smooth = X_smooth.reshape((len(X_smooth), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
#use poly_reg.fit_transform(X) so that lin_reg_2 can be used for predicting
#any new matrix of X
plt.plot(X_smooth, lin_reg_2.predict(poly_reg.fit_transform(X_smooth)), color = 'green')
plt.title('Truth or Bluff')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


"""
Step 7: Predicting a new result with Linear and Polynomial Regression
"""
lin_reg.predict(6.5)
lin_reg_2.predict(poly_reg.fit_transform(6.5))






"""
Step 4: Feature Scaling: Standardization and Normalization

Standardization equation:
   Xstand = (X - mean(X)) / (standard deviation(X))

Normalization equation:
   Xnorm = (X - min(X)) / (max(X) - min(X))

Note: most ML libraries deal with feature scaling. But some do not


#import StandardScaler 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
"""















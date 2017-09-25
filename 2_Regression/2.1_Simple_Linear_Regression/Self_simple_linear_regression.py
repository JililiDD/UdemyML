#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 21:05:52 2017

@author: ddxy
"""

#Simple Linear Regression using OLS(Ordinary Least Squares)

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

dataset = pd.read_csv('Salary_Data.csv')

#2.2: create a matrix of features/IVs
X = dataset.iloc[:,:-1].values #[:,:-1] means select all the rows and the 
                               #columns except the last column

#2.3: create a dependent variable vector
y = dataset.iloc[:,1].values


"""
Step 3: Splitting the dataset into the training and test sets
"""
from sklearn.cross_validation import train_test_split
#random_state = 0 is to make everytime it gives the same result. it is like
#specifying a seed for random() in java
#practically, we do not need this so that every time the program will
#randomly select 1/3 of data for the test.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)


"""
Step 4: Fitting Simple Linear Regression of the Training set
"""
#import LinearRegression class from linear_model library to create an object to
#create a model using its methods
from sklearn.linear_model import LinearRegression
regressor = LinearRegression() #press control + i can get the details of the
                               #class- LinearRegression()
                               
#fit traning set to the regressor object
"""
**************************************************************************************
#Using the machine learning concept. The "regressor" above in this case corresponds
#to the "machine". The "fit()" method below in this case corresponds to the 
#"learning"
**************************************************************************************
"""
regressor.fit(X_train, y_train)


"""
Step 5: Show statistical results of using OLS on the training set
"""
from statsmodels.formula.api import ols
#split the dataset to training and test sets
train_set, test_set = train_test_split(dataset, test_size = 1/3, random_state = 0)
#model = ols("Salary ~ YearsExperience", train_set).fit()
#print(model.summary())

#create the formula string
tempHeaders = list(dataset)
formula = tempHeaders[-1] + " ~ "
for iv in range(0, len(tempHeaders) - 1):
    formula += tempHeaders[iv] + " + "

formula = formula[:-3]

#use ols to create the model
model = ols(formula, train_set).fit()
print(model.summary())


"""
Step 6: Predicting the Test set results
"""
#Use the regressor object to predict predicted y according to the X_test set
y_pred = regressor.predict(X_test)

#Return MSE of the test set using "regressor" model
print ("Fit a model X_train, and calculate MSE with X_test, Y_test: ")
np.mean((y_test - y_pred) ** 2)



"""
Step 7: Visualising the Training set results
"""
#use matplotlib.pyplot to graph the training set plot
#Step 7.1: create a scatter plot of the original training set data
plt.scatter(X_train, y_train, color = 'red')

#Step 7.2: plot the fitted regression line to the scatter plot by calculating
#the predicted y points according to the X_train data points 
plt.plot(X_train, regressor.predict(X_train), color = 'blue')

#Step 7.3: add title and the labels for x-axis and y-axis
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')

#Step 7.4: use show() at the end to show the plot
plt.show()


"""
Step 8: Visualising the Test set results
"""
#use matplotlib.pyplot to graph the Test set plot
#Step 8.1: create a scatter plot of the original test set data
plt.scatter(X_test, y_test, color = 'red')

#Step 8.2: plot the modeled regression line to the scatter plot
#NOTE that 
#       plt.plot(X_test, regressor.predict(X_test), color = 'blue')
#will produce the same fitted line because regressor object is ALREADY TRAINED
#the regressor is already a MODEL that is used for prediction. Therefore,
#both X_train and X_test will produce the same fitted line by being applied the
#same model -- regressor in this case

plt.plot(X_train, regressor.predict(X_train), color = 'blue')

#Step 8.3: add title and the labels for x-axis and y-axis
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')

#Step 8.4: use show() at the end to show the plot
plt.show()









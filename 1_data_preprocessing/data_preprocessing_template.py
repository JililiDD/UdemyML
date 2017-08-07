"""
    This is for Data Preprocessing
    
    
Original dataset:
    Country	Age	Salary	Purchased
    France	   44	   72000	No
    Spain	   27	   48000	Yes
    Germany	30	   54000	No
    Spain	   38	   61000	No
    Germany	40		        Yes
    France	   35	   58000	Yes
    Spain		      52000	No
    France	   48	   79000	Yes
    Germany	50	   83000	No
    France	   37	   67000	Yes

"""

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

dataset = pd.read_csv('Data.csv')

#2.2: create a matrix of features/IVs
X = dataset.iloc[:,:-1].values #[:,:-1] means select all the rows and the 
                               #columns except the last column

#2.3: create a dependent variable vector
y = dataset.iloc[:,3].values


"""
Step 3: Splitting the dataset into the training and test sets
"""
from sklearn.cross_validation import train_test_split
#random_state = 0 is to make everytime it gives the same result. it is like
#specifying a seed for random() in java
#practically, we do not need this so that every time the program will
#randomly select 20% of data for the test.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)





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








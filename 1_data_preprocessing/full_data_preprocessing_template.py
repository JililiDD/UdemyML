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
Step 3: data cleaning: deal with missing data and create dummy variables
"""
#3.1 taking care of missing data (fill in the blanks with means of the corresponding column)
#import a library to deal with this instead of writting the function ourselves
from sklearn.preprocessing import Imputer

#create an Impute object. Right after typing "Imputer", we can use ctrl+i to
#bring up the documentation to see what parameters we need for creating a 
#correct Imputer object.
#axis = 0 means take the mean based on the column (row if axis = 1)
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
#fit the columns in X that needs to be processed into imputer
#fit() calculates some descriptive stats for further scaling. These descriptive 
#stats are stored internally in the object as states. fit() doesn't return 
#anything. 
imputer.fit(X[:,1:3])

#from imputer, transform the processed data back to the X
#transform() uses the calculated descriptive stats from fit() to transform and
#RETURN a transformed dataset. Such transformation can be standardization
#and normalization. 
#fit_transform() combines the fit() and transform() methods together
X[:,1:3] = imputer.transform(X[:, 1:3])


#3.2 Encoding categorical data with dummy variables

#3.2.1 
#first convert the strings in the first column to integer coding
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])


#3.2.2 
#after 2.5.1 the first column is coded with 0,1,2. The problem with
#this is that the categorical variable should not have numerical meaning 
#after conversion. However, 0,1,2 has numerical meaning like 2>1>0.
#this is not good for analysis. Therefore, we need to fuuther convert this
#column into dummy variables. In this case, each country in this column will
#create a new column after the conversion

#Output after conversion 
#France	Germany	Spain	 Age	Salary
#     1	     0	    0	 44	72000
#     0	     0	    1	 27	48000
#     0	     1	    0	 30	54000
#     0	     0	    1	 38	61000
#     0	     1	    0	 40	63778
#     1	     0	    0	 35	58000
#     0	     0	    1	 39	52000
#     1	     0	    0	 48	79000
#     0	     1	    0	 50	83000
#     1	     0	    0	 37	67000



onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

#3.2.3: 
#encode the y column
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)


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
Step 5: Feature Scaling: Standardization and Normalization

Standardization equation:
   Xstand = (X - mean(X)) / (standard deviation(X))

Normalization equation:
   Xnorm = (X - min(X)) / (max(X) - min(X))
"""
#import StandardScaler 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)









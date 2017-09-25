"
#This tutorial is for Data preprocessing 

Original dataset:
    Country	Age	   Salary	Purchased
    France	 44	   72000	No
    Spain	   27	   48000	Yes
    Germany	 30	   54000	No
    Spain	   38	   61000	No
    Germany	 40		        Yes
    France	 35	   58000	Yes
    Spain		       52000	No
    France	 48	   79000	Yes
    Germany	 50	   83000	No
    France	 37	   67000	Yes
"

#step 1: import the dataset
#set a working directory by going to the Files pane, find the correct folder, then go
#to More and click on "Set as Working Directory"
dataset = read.csv('Data.csv')
#dataset = dataset[, 2:3]


#Step 2: Splitting the dataset into the Training set and Test set
#install caTools library by install.packages('caTools')
#Then, check caTools in the Packages pane. OR use
library(caTools)
set.seed(123) #set seed so that every time we get the same random result

#sample.split() will randomly select specified percentage of tuples to be the training
#set and label them as TRUE. The test set tuples are labeled as FALSE
split = sample.split(dataset$Purchased, SplitRatio = 0.8)

#create traning set and test set based on the TRUE and FALSE label created in the 
#last step in dataset
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

#Feature scaling
# training_set[, 2:3] = scale(training_set[, 2:3])
# test_set[, 2:3] = scale(test_set[, 2:3])








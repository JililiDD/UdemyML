#Multiple Linear Regression

#step 1: import the dataset
#set a working directory by going to the Files pane, find the correct folder, then go
#to More and click on "Set as Working Directory"
dataset = read.csv('50_Startups.csv')

#Step 2: Convert categorical data to dummy variables
dataset$State = factor(dataset$State,
                         levels = c('California', 'Florida', 'New York'), #c() means vector in R
                         labels = c(1, 2, 3))

#Step 3: Splitting the dataset into the Training set and Test set
#install caTools library by install.packages('caTools')
#Then, check caTools in the Packages pane. OR use
library(caTools)
set.seed(123) #set seed so that every time we get the same random result

#sample.split() will randomly select specified percentage of tuples to be the training
#set and label them as TRUE. The test set tuples are labeled as FALSE
split = sample.split(dataset$Profit, SplitRatio = 0.8)

#create traning set and test set based on the TRUE and FALSE label created in the 
#last step in dataset
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

#Feature scaling
#training_set[, 2:3] = scale(training_set[, 2:3])
#test_set[, 2:3] = scale(test_set[, 2:3])

#Step 4: create the model
#NOTE that R will automatically remove one of the dummy variables to fulfill
#the "independence of errors" assumption
regressor = lm(formula = Profit ~ ., # OR: formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State,
               data = training_set
               )
summary(regressor)
#show the coefficients of each terms in the model
coef(regressor)

#Step 5: Predicting the Test set results
y_pred = predict(regressor, newdata = test_set)







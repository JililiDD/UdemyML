"
Simple Linear Regression
"

#step 1: import the dataset
#set a working directory by going to the Files pane, find the correct folder, then go
#to More and click on "Set as Working Directory"
dataset = read.csv('Salary_Data.csv')
#dataset = dataset[, 2:3]


#Step 2: Splitting the dataset into the Training set and Test set
#install caTools library by install.packages('caTools')
#Then, check caTools in the Packages pane. OR use
library(caTools)
set.seed(123) #set seed so that every time we get the same random result

#sample.split() will randomly select specified percentage of tuples to be the training
#set and label them as TRUE. The test set tuples are labeled as FALSE
split = sample.split(dataset$Salary, SplitRatio = 2/3)

#create traning set and test set based on the TRUE and FALSE label created in the 
#last step in dataset
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

#Feature scaling
# training_set[, 2:3] = scale(training_set[, 2:3])
# test_set[, 2:3] = scale(test_set[, 2:3])


#Step 3: Fit the simple linear regression to the Training set
#Salary ~ YearsExperience indicates the DV and the IV respectively
regressor = lm(formula = Salary ~ YearsExperience, 
               data = training_set
               )
#show statistical results of the regressor
summary(regressor)
#show the coefficients of each terms in the model
coef(regressor)

#Step 4: Predicting the Test set result
y_pred = predict(regressor, newdata = test_set)

#Step 5: Using ggplot2 to visualise the Training set results

#install ggplot2
#install.packages('ggplot2')

library(ggplot2)

ggplot() +
  geom_point(aes(x = training_set$YearsExperience, y = training_set$Salary),
             colour = 'red') +
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
             colour = 'blue') +
  ggtitle('Salary vs Experience (Training set)') +
  xlab('Years of experience') +
  ylab('Salary')

#Step 6: Using ggplot2 to visualise the Test set results

#install ggplot2
#install.packages('ggplot2')

library(ggplot2)

ggplot() +
  geom_point(aes(x = test_set$YearsExperience, y = test_set$Salary),
             colour = 'red') +
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
            colour = 'blue') +
  ggtitle('Salary vs Experience (Test set)') +
  xlab('Years of experience') +
  ylab('Salary')





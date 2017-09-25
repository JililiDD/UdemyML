#Polynomial Regression

#step 1: import the dataset
#set a working directory by going to the Files pane, find the correct folder, then go
#to More and click on "Set as Working Directory"
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]

#step 2: Fitting Linear Regression to the dataset
lin_reg = lm(formula = Salary ~ . ,
             data = dataset)
summary(lin_reg)
#show the coefficients of each terms in the model
coef(lin_reg)

#step 3: Fitting Polynomial Regression to the dataset
dataset$Level2 = dataset$Level^2
dataset$Level3 = dataset$Level^3
dataset$Level4 = dataset$Level^4
poly_reg = lm(formula = Salary ~ . ,
              data = dataset)
summary(poly_reg)
#show the coefficients of each terms in the model
coef(poly_reg)

#Step 4: Visualising the Linear and Polynomial Regression results
library(ggplot2)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary), colour = 'red') +
  geom_line(aes(x = dataset$Level, y = predict(lin_reg, newdata = dataset)), colour = 'blue') +
  geom_line(aes(x = dataset$Level, y = predict(poly_reg, newdata = dataset)), colour = 'green') +
  ggtitle('Truth or Bluff') +
  xlab('Level') +
  ylab('Salary')

#Step 5: Predicting a new result with Linear and Polynomial Regression
#for a single number, we need to create a data frame (a cell in this case) 
y_lin_pred = predict(lin_reg, data.frame(Level = 6.5))

#for polynomial model, we need to create a data frame to include all the terms in the model
y_poly_pred = predict(poly_reg, data.frame(Level = 6.5, 
                                           Level2 = 6.5^2,
                                           Level3 = 6.5^3,
                                           Level4 = 6.5^4))

















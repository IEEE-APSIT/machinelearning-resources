# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math


# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')#has only 2 columns experience and salary(index 0 and 1)

X = dataset.iloc[:, :-1].values	#takes all column except last column
#(here it takes experience column as input which has index 0)

y = dataset.iloc[:, 1].values	#Takes column with index 1 as input
#(here its the salary column) or you can use the below command also

#y = dataset.iloc[:, -1].values #Takes only last column ie. salary column

# Splitting the dataset into the Training set and Test set
#sklearn.cross_validation is now depreceated.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)


print("Experience    Predicted Salary   Actual Salary")
for i in range(0,7):
    print(X_test[i],"       ",math.trunc(y_pred[i]),"          ",math.trunc(y_test[i]))
    
#math.trunc decimal values ko whole kardeta hai aur kn
    
    
# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


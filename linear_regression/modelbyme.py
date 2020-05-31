import numpy as np
import pandas as pd

df=pd.read_csv('50_Startups.csv')

if True in df.isnull():
    print("missing value detected")
else:
    print("No missing value")    
    
#Basic data visualization for dataset

print(df.columns)
print(df.dtypes)
print(df.head)
print(df.corr())    
    
    
#X is for depedent/input variables which are from column o to 3
#y is for independent/output variable which is in last column number 4 as per indexing    
    
X=df.iloc[:,:-1].values #takes all values as input except last one
y=df.iloc[:,-1].values    #takes only last value as input
    

#Now we have to encode categorical data and it can be done by 2 ways
#sklearn 0.20 has categorical_features class while next versions have ColumnTransformer

"""# Encoding categorical data using categorical_features
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()"""


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[3])],remainder='passthrough')
X=np.array(ct.fit_transform(X))


#in variable explorer u see x is of object type and has lots of decimal values
X=X.astype(int) #now it has no decimal values

print(X)
#now compare values inside X in ipython console and df from variabe explorer
#all column values are same except in the start we have 3 newcolumns now instead of state column
#on comparing we get 3rd column is for NewYork,1st for california and 2nd for Florida

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)



# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
print("Predicted values for corresponding values in x_test are stored in y_pred while actual values are stored in y_test")

print("   Parameters                                  Predicted Score       Actual Score ")
for i in range(0,9):
    print(X_test[i],"  ",y_pred[i],"  ",y_test[i])
import pandas as pd
import numpy as np

data = pd.read_csv('G:/Github projects/LinearRegression/Multiple Linear Regression/Scikit Implementation/bostonhousingdata.csv')

X = data.iloc[:, :-1].values # All features except PRICE
y = data.iloc[:, -1].values # PRICE
print(X.shape)

# Creating model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X, y)

# Making predictions
testpoint = np.array([[0.67, 12.5, 8.2, 1, 0.5, 7.79, 89.88, 5.13, 4, 320, 20.8, 412.5, 17.63]])
testpred = model.predict(testpoint)
print("Test point prediction: ", testpred)


# Coefficients and intercepts
intercept = model.intercept_
print("intercept: ", intercept)

columns = data.columns
coefficients = pd.DataFrame(model.coef_, columns[:-1], columns=['Coefficients'])
print("\n", coefficients)


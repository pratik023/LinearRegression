import numpy as np 
import pandas as pd 


def gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)
    
    for i in range(num_iters):
        error = np.dot(X, theta) - y
        theta = theta - ((alpha/m) * ((np.dot(error.T, X)).T))

    return theta


data = pd.read_csv('G:/Github projects/LinearRegression/Multiple Linear Regression/From Scratch/bostonhousingdata.csv')
print(data.head())

X = data.iloc[:, :-1].values # All features except PRICE
y = data.iloc[:, -1].values # PRICE
y = y.reshape(len(y), 1)

X = np.c_[np.ones(len(y)), X]
print(X, X.shape[1])

# Setting the parameters
alpha = 1.0e-7
iterations = 5000
theta = np.zeros((X.shape[1],1))

theta = gradient_descent(X, y, theta, alpha, iterations)
print(theta)

# Testpoint
testpoint = np.array([[1, 0.67, 12.5, 8.2, 1, 0.5, 7.79, 89.88, 5.13, 4, 320, 20.8, 412.5, 17.63]])
prediction = np.dot(testpoint, theta)
print("Test point prediction: ", prediction)


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt


def gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)
   
    for _ in range(num_iters):
        error = np.dot(X, theta) - y
        t1 = theta[0] - ((alpha/m) * np.sum(np.multiply(error, X[:,0].reshape(m,1))))
        t2 = theta[1] - ((alpha/m) * np.sum(np.multiply(error, X[:,1].reshape(m,1))))
        
        theta = np.array([t1,t2])
        
    return theta


# Reading data
data = pd.read_csv('Salary_Data.csv')

X = data['YearsExperience']
y = data['Salary']

# print(X, y)
# print(X.shape, y.shape)

# Data visualization
plt.scatter(X, y, c='r')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.title('Linear Regression')
plt.show()

# Adding an extra column of ones
X = np.c_[np.ones(len(y)), X]

y = y.values.reshape(len(y), 1)


# Setting parameters
alpha = 0.01
iterations = 2500
theta = np.zeros((2,1))

grad_result = gradient_descent(X, y, theta, alpha, iterations)
# print(grad_result)

y_intercept = grad_result[0][0]
slope = grad_result[1][0]

print("\nEquation of line y = {}x + {}".format(slope, y_intercept))

testpoint = [1, 2.3]
prediction = np.dot(testpoint, grad_result)
print("Prediction for 2.3 years of prediction: ", prediction[0])


# Visualizing the best fit line
plt.scatter(X[:,1], y)
plt.plot(X[:,1], np.dot(X, grad_result), c='r')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.title('Linear Regression')
plt.show()

'''
OUTPUT:

Equation of line y = 9467.7512866598x + 25672.325083633412
Prediction for 2.3 years of prediction:  47448.15304295095

'''



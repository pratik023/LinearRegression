import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("LinearRegression/Scikit Implementation/Salary_Data.csv")
print(data)

X = data[['YearsExperience']]
y = data['Salary']

# Data Visualization
plt.scatter(X, y)
plt.xlabel('Years Experience')
plt.ylabel('Salary')
plt.title('Linear Regression')
plt.show()

print(X.shape, y.shape)

# Model Building
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X, y)

testdata = np.array([[2.3]])
prediction = model.predict(testdata)
print("Prediction for 2.3 years of prediction: ", prediction[0])

slope = model.coef_[0]
y_intecept = model.intercept_
print("\nEquation of line y = {}x + {}".format(slope, y_intecept))


# Plotting the best fit line
regressionline = []
for x in data['YearsExperience']:
    regressionline.append((slope*x) + y_intecept)

plt.scatter(X, y)
plt.plot(data['YearsExperience'], regressionline, c='r')
plt.xlabel('Years Experience')
plt.ylabel('Salary')
plt.title('Linear Regression')
plt.show()

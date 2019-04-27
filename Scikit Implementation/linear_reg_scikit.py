import numpy as np 
import pandas as pd 

data = pd.read_csv("LinearRegression/Scikit Implementation/Salary_Data.csv")
print(data)

X = data[['YearsExperience']]
y = data['Salary']

print(X.shape, y.shape)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X, y)

testdata = np.array([[2.3]])
prediction = model.predict(testdata)
print("Prediction for 2.3 years of prediction: ", prediction[0])
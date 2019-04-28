import pandas as pd
import numpy as np

data = pd.read_csv('G:/Github projects/LinearRegression/Multiple Linear Regression/Scikit Implementation/bostonhousingdata.csv')

X = data.iloc[:, :-1].values # All features except PRICE
y = data.iloc[:, -1].values # PRICE
print(X.shape)

# Creating training and testing datasets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# Creating model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

# Making predictions
predictions = model.predict(X_test)
print(predictions)

from sklearn import metrics
print("\nRoot mean squared Error (RMSE): ", np.sqrt(metrics.mean_squared_error(y_test, predictions)))

testpoint = np.array([[0.67, 12.5, 8.2, 1, 0.5, 6.79, 89.88, 5.43, 4, 320, 20.8, 348.5, 13.33]])
testpred = model.predict(testpoint)
print("Test point prediction: ", testpred)

# Coefficients
columns = data.columns
coefficients = pd.DataFrame(model.coef_, columns[:-1], columns=['Coefficients'])
print("\n", coefficients)


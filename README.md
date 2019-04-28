# LinearRegression
#### Linear Regression in Python - Scikit and Scratch

#### Scikit Implementation
This is a very basic implementation of Linear Regression for beginners. It includes reading a dataset and using it to build a model
This model is then used to predict outcome for test point. </br>
It involves some visualization of data.

#### Scratch Implementation

This code unfolds the logic behind the working of Linear Regression. The scikit code is very abstract, easy to implement for beginners but doesn't give enough idea about what's happening at the backend.  </br>
This code implements Linear Regression using the 'Gradient Descent Approach' for updating the weights.
All the operations are done in the form of matrix multiplications.</br>
It also has basic visualizations of data.

## 1. Simple Linear Regression
Simple Linear Regression establishes relation between two variables. One is called as a <b>'dependent variable'</b> and the other is called as <b>'independent variable'</b>. </br>
The dependent variable is used to predict the value of independent variable. </br>
The model is represented as: <b>y = mx + c</b> </br>
where, y - independent variable</br>
       x - dependent variable</br>
       c - y_intercept (i.e. value of y when x=0)


## 2. Multiple Linear Regression
The basic difference between Simple and Multiple Linear Regression is that the later one deals with multiple 'dependent variables' and a single 'independent variable'. </br>
The model is represented as: y = m<sub>1</sub>x<sub>1</sub> + m<sub>2</sub>x<sub>2</sub> + .... + m<sub>n</sub>x<sub>n</sub> </br></br>

The Scikit implementation though is as usual simple but doesn't give a lot of insights of what happens in the background.</br>
The scratch implementation unfolds the working of gradient descent, however it is not the very best implementation of multiple linear regression. <br/>
The model built in the scratch code requires a better analysis and some advanced techniques to improve its performance and predict better.
But for beginners, it gives the essential idea of how multiple linear regression is different from simple regression.


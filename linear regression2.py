from sklearn.linear_model import LinearRegression
import numpy as np


X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)  
y = np.array([2, 4, 5, 4, 5])  

model = LinearRegression()


model.fit(X, y)


slope = model.coef_[0]
intercept = model.intercept_


print("Slope (Coefficient):", slope)
print("Y-intercept (Intercept):", intercept)


x_new = np.array([6]).reshape(-1, 1)  
y_pred = model.predict(x_new)

print("Predicted value for x =", x_new[0][0], "is y =", y_pred[0])

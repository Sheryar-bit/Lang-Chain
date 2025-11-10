## Linear Regression basic code example
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression

# Sample data
X = np.array([[1], [2], [3], [4], [5]])   # number of rooms
y = np.array([160, 210, 260, 300, 400])   # price in thousands (Numeric means its Linear regression)

# Create & train model
model = LinearRegression()
model.fit(X, y)

# Make prediction
pred = model.predict([[6]])   # predict price for 6 rooms
print(f"Predicted price for 6 rooms: ${pred[0]:.2f}k")

# Visualize
plt.scatter(X, y, color="blue", label="Actual data")
plt.plot(X, model.predict(X), color="red", label="Regression line")
plt.xlabel("Number of rooms")
plt.ylabel("Price (in $1000s)")
plt.legend()
plt.show()


## Linear Classification basic code example

# Sample data
X = np.array([[30], [40], [50], [60], [70], [80], [90]])
y = np.array([0, 0, 0, 1, 1, 1, 1])   # 0 = Fail, 1 = Pass (we have discrete categories)

# Train classifier
clf = LogisticRegression()
clf.fit(X, y)

# Predict
pred = clf.predict([[55]])
print(f"Prediction for 55 marks: {'Pass' if pred[0] == 1 else 'Fail'}")

# Visualize
x_test = np.linspace(20, 100, 200).reshape(-1, 1)
prob = clf.predict_proba(x_test)[:, 1]  # probability of class 1 (Pass)

plt.scatter(X, y, color="blue", label="Actual data")
plt.plot(x_test, prob, color="red", label="Decision boundary (sigmoid curve)")
plt.xlabel("Marks")
plt.ylabel("Probability of Passing")
plt.legend()
plt.show()

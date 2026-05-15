import numpy as np
import matplotlib.pyplot as plt

# AND gate data
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,0,0,1])

# Initialize
w = np.zeros(2)
b = 0
lr = 0.1

# Training
for _ in range(10):
    for i in range(len(X)):
        y_pred = 1 if np.dot(X[i], w) + b >= 0 else 0
        error = y[i] - y_pred
        w += lr * error * X[i]
        b += lr * error

# Decision region
xx, yy = np.meshgrid(np.arange(-0.5,1.5,0.01),
                     np.arange(-0.5,1.5,0.01))

Z = np.where(np.dot(np.c_[xx.ravel(), yy.ravel()], w) + b >= 0, 1, 0)
Z = Z.reshape(xx.shape)

# Plot
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
plt.scatter(X[:,0], X[:,1], c=y, s=100, cmap='coolwarm')

plt.title("Perceptron Decision Region")
plt.xlabel("X1")
plt.ylabel("X2")
plt.grid()
plt.show()
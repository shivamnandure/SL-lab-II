import numpy as np
import matplotlib.pyplot as plt

# XOR dataset
X = np.array([[0,0],[0,1],[1,0],[1,1]])
Y = np.array([[0],[1],[1],[0]])

# Sigmoid function
sigmoid = lambda x: 1/(1+np.exp(-x))
derivative = lambda x: x*(1-x)

# Random weights
np.random.seed(1)
w1 = np.random.rand(2,2)
w2 = np.random.rand(2,1)

errors = []

# Training
for i in range(10000):

    # Forward propagation
    h = sigmoid(np.dot(X,w1))
    o = sigmoid(np.dot(h,w2))

    # Error
    e = Y-o
    errors.append(np.mean(np.abs(e)))

    # Backpropagation
    d_o = e * derivative(o)
    d_h = d_o.dot(w2.T) * derivative(h)

    # Weight update
    w2 += h.T.dot(d_o) * 0.1
    w1 += X.T.dot(d_h) * 0.1

# Output
print("Predicted Output:\n",o)

# Graph
plt.plot(errors)
plt.title("Error Reduction")
plt.xlabel("Epochs")
plt.ylabel("Error")
plt.grid()
plt.show()
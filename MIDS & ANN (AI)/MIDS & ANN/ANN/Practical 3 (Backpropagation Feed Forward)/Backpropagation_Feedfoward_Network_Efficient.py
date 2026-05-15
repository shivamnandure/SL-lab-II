import numpy as np
import matplotlib.pyplot as plt

# XOR Dataset
X = np.array([[0,0],[0,1],[1,0],[1,1]])
Y = np.array([[0],[1],[1],[0]])

# Sigmoid Function
sig = lambda x: 1/(1+np.exp(-x))

# Random Weights
wh = np.random.rand(2,2)
wo = np.random.rand(2,1)

lr = 0.1
errors = []

# Training
for i in range(10000):

    # Forward Propagation
    h = sig(X.dot(wh))
    o = sig(h.dot(wo))

    # Error
    e = Y - o
    errors.append(np.mean(e**2))

    # Back Propagation
    do = e * o * (1-o)
    dh = do.dot(wo.T) * h * (1-h)

    # Weight Update
    wo += h.T.dot(do) * lr
    wh += X.T.dot(dh) * lr

# Output
print("Predicted Output:\n", o)

# Graph
plt.plot(errors)
plt.xlabel("Epochs")
plt.ylabel("Error")
plt.title("Back Propagation Learning")
plt.show()
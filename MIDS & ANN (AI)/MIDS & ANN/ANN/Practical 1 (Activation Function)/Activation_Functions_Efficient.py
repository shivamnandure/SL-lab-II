# Activation Functions in Neural Networks

import numpy as np
import matplotlib.pyplot as plt

# Input values for graphs
x = np.linspace(-10, 10, 200)

# ---------------- Sigmoid ----------------
sigmoid = 1 / (1 + np.exp(-x))

plt.figure()
plt.plot(x, sigmoid)
plt.title("Sigmoid Function")
plt.grid()
plt.show()

# ---------------- ReLU ----------------
relu = np.maximum(0, x)

plt.figure()
plt.plot(x, relu)
plt.title("ReLU Function")
plt.grid()
plt.show()

# ---------------- Tanh ----------------
tanh = np.tanh(x)

plt.figure()
plt.plot(x, tanh)
plt.title("Tanh Function")
plt.grid()
plt.show()

# ---------------- Leaky ReLU ----------------
leaky_relu = np.where(x > 0, x, 0.01 * x)

plt.figure()
plt.plot(x, leaky_relu)
plt.title("Leaky ReLU Function")
plt.grid()
plt.show()

# ---------------- Softmax ----------------
arr = np.array([1, 2, 3, 4])

softmax = np.exp(arr) / np.sum(np.exp(arr))

plt.figure()
plt.bar(range(len(arr)), softmax)
plt.title("Softmax Function")
plt.xlabel("Input Index")
plt.ylabel("Probability")
plt.grid()
plt.show()

# Display Values
print("\nSoftmax Output =", softmax)
print("\nSigmoid     =", sigmoid)
print("\nReLU        =", relu)
print("\nTanh        =", tanh)
print("\nLeaky ReLU  =", leaky_relu)
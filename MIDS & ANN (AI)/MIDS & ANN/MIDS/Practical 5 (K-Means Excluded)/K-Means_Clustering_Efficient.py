# K-Means Clustering (Efficient Practical Code)

import numpy as np
import matplotlib.pyplot as plt

# Points
P = {
    'P1':[0.1,0.6], 'P2':[0.15,0.71], 'P3':[0.08,0.9], 'P4':[0.16,0.85],
    'P5':[0.2,0.3], 'P6':[0.25,0.5], 'P7':[0.24,0.1], 'P8':[0.3,0.2]
}

# Initial Centroids
m1, m2 = np.array(P['P1']), np.array(P['P8'])

C1, C2 = [], []

# Cluster Formation
for name, point in P.items():

    point = np.array(point)

    d1 = np.linalg.norm(point - m1)   # Distance from m1
    d2 = np.linalg.norm(point - m2)   # Distance from m2

    if d1 < d2:
        C1.append((name, point))
    else:
        C2.append((name, point))

# Updated Centroids
m1_new = np.mean([p[1] for p in C1], axis=0)
m2_new = np.mean([p[1] for p in C2], axis=0)

# Final Answers
print("Cluster C1 :", [p[0] for p in C1])
print("Cluster C2 :", [p[0] for p in C2])

print("\n1] P6 belongs to :", "C1" if any(p[0]=='P6' for p in C1) else "C2")
print("2] Population around m2 :", len(C2))

print("\n3] Updated m1 =", m1_new)
print("   Updated m2 =", m2_new)

# Visualization
for name, point in C1:
    plt.scatter(point[0], point[1])
    plt.text(point[0], point[1], name)

for name, point in C2:
    plt.scatter(point[0], point[1])
    plt.text(point[0], point[1], name)

plt.scatter(m1_new[0], m1_new[1], marker='X', s=200, label='m1')
plt.scatter(m2_new[0], m2_new[1], marker='X', s=200, label='m2')

plt.title("K-Means Clustering")
plt.legend()
plt.grid()
plt.show()
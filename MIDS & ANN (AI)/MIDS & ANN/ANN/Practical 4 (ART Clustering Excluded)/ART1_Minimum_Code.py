# ART1 - Adaptive Resonance Theory

import numpy as np
import matplotlib.pyplot as plt

# Input patterns
p = np.array([[1,1,0,0],
              [1,0,0,0],
              [0,0,1,1],
              [0,0,1,0]])

v = 0.6          # Vigilance parameter
c = []           # Clusters
labels = []      # Cluster numbers

# Similarity function
def sim(a,b):
    return np.sum(a & b) / np.sum(a)

# ART1 clustering
for x in p:

    done = False

    for i,y in enumerate(c):

        if sim(x,y) >= v:
            c[i] = x & y
            labels.append(i+1)
            done = True
            break

    if not done:
        c.append(x.copy())
        labels.append(len(c))

# Output
print("Cluster Assignment:", labels)

# Visualization
plt.bar(range(1,len(labels)+1), labels)
plt.xlabel("Pattern Number")
plt.ylabel("Cluster Number")
plt.title("ART1 Clustering")
plt.show()

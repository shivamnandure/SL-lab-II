# =====================================================
# ASSIGNMENT 4 : ABALONE DATASET
# a) Ring Prediction (Classification)
# b) Age Prediction (Linear Regression)
# =====================================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score

# =====================================================
# LOAD DATASET
# =====================================================

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"

cols = [
    "Sex","Length","Diameter","Height",
    "Whole_weight","Shucked_weight",
    "Viscera_weight","Shell_weight","Rings"
]

df = pd.read_csv(url, names=cols)

print(df.head())

# =====================================================
# PREPROCESSING
# =====================================================

# Convert categorical values to numeric
df["Sex"] = LabelEncoder().fit_transform(df["Sex"])

# =====================================================
# VISUALIZATIONS
# =====================================================

# Histogram
df.hist(figsize=(10,8))
plt.show()

# Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.show()

# Scatter Plot
plt.scatter(df["Shell_weight"], df["Rings"])
plt.xlabel("Shell Weight")
plt.ylabel("Rings")
plt.title("Shell Weight vs Rings")
plt.show()

# =====================================================
# PART A : CLASSIFICATION
# Predict Number of Rings
# =====================================================

X = df.drop("Rings", axis=1)
y = df["Rings"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Random Forest Classifier
clf = RandomForestClassifier()

clf.fit(X_train, y_train)

pred = clf.predict(X_test)

print("\nClassification Accuracy =", accuracy_score(y_test, pred))

# =====================================================
# PART B : LINEAR REGRESSION
# Predict Age
# Age = Rings + 1.5
# =====================================================

df["Age"] = df["Rings"] + 1.5

X = df.drop(["Rings", "Age"], axis=1)
y = df["Age"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Linear Regression Model
model = LinearRegression()

model.fit(X_train, y_train)

pred_age = model.predict(X_test)

# Evaluation
print("\nMean Absolute Error =", mean_absolute_error(y_test, pred_age))

print("R2 Score =", r2_score(y_test, pred_age))

# =====================================================
# REGRESSION VISUALIZATION
# =====================================================

plt.scatter(y_test, pred_age)

plt.xlabel("Actual Age")

plt.ylabel("Predicted Age")

plt.title("Actual vs Predicted Age")

plt.show()

# =====================================================
# USER INPUT PREDICTION
# =====================================================

print("\nEnter Abalone Details")

sex = input("Sex (M/F/I): ")
length = float(input("Length: "))
diameter = float(input("Diameter: "))
height = float(input("Height: "))
whole = float(input("Whole Weight: "))
shucked = float(input("Shucked Weight: "))
viscera = float(input("Viscera Weight: "))
shell = float(input("Shell Weight: "))

# Encode Sex
sex = {"M":2, "F":0, "I":1}[sex]

sample = np.array([[
    sex, length, diameter, height,
    whole, shucked, viscera, shell
]])

# Predictions
rings = clf.predict(sample)
age = model.predict(sample)

print("\nPredicted Rings =", rings[0])

print("Predicted Age =", age[0])
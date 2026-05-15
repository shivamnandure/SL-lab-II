# Assignment 3 - Sentiment Analysis using Naive Bayes

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Dataset
data = {
    "Text": [
        "I love this movie", "Amazing product", "Very happy",
        "Worst experience", "I hate this", "Very bad service"
    ],
    "Sentiment": [
        "Positive", "Positive", "Positive",
        "Negative", "Negative", "Negative"
    ]
}

df = pd.DataFrame(data)

# Graph 1 - Sentiment Distribution
sns.countplot(x="Sentiment", data=df)
plt.title("Sentiment Distribution")
plt.show()

# Convert Text into Numbers
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(df["Text"])
y = df["Sentiment"]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train Model
model = MultinomialNB()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Graph 2 - Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=["Negative", "Positive"],
    yticklabels=["Negative", "Positive"]
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

# User Input Prediction
while True:

    text = input("\nEnter sentence (type exit to stop): ")

    if text.lower() == "exit":
        break

    test = vectorizer.transform([text])

    result = model.predict(test)

    print("Predicted Sentiment:", result[0])

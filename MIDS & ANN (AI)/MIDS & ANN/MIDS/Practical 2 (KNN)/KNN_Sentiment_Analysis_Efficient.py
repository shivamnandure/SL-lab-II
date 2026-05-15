# =========================================
# SENTIMENT ANALYSIS USING KNN
# =========================================

# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# =========================================
# Load Dataset
# =========================================

df = pd.read_csv("tweets _dataset.csv", encoding='latin1')

# Features and Labels
X = df['text']
y = df['sentiment']

# =========================================
# Text Vectorization using TF-IDF
# =========================================

tfidf = TfidfVectorizer(stop_words='english')
X = tfidf.fit_transform(X)

# =========================================
# Split Dataset
# =========================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================================
# Train KNN Model
# =========================================

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# =========================================
# Predictions and Accuracy
# =========================================

y_pred = model.predict(X_test)

print("\nAccuracy :", accuracy_score(y_test, y_pred))
print("\nClassification Report :\n")
print(classification_report(y_test, y_pred))

# =========================================
# Confusion Matrix Visualization
# =========================================

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# =========================================
# Custom Tweet Prediction
# =========================================

tweet = ["I love this product"]

tweet_vector = tfidf.transform(tweet)

print("\nTweet :", tweet[0])
print("Predicted Sentiment :", model.predict(tweet_vector)[0])

# =========================================
# END
# =========================================
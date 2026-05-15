# ============================================
# TITANIC DATA PREPROCESSING
# ============================================

# Import Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load Dataset
df = pd.read_csv("titanic_dataset.csv")

# Display Dataset
print(df.head())
print(df.info())

# Check Missing Values
print(df.isnull().sum())

# Visualizations
sns.countplot(x='Survived', data=df)
plt.title("Survival Count")
plt.show()

sns.histplot(df['Age'], kde=True)
plt.title("Age Distribution")
plt.show()

sns.boxplot(x=df['Fare'])
plt.title("Fare Boxplot")
plt.show()

# Handle Missing Values
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop Unnecessary Columns
df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Convert Categorical Data
df['Sex'] = df['Sex'].map({'male':0, 'female':1})
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# Remove Duplicate Values
df.drop_duplicates(inplace=True)

# Feature Scaling
scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

# Correlation Heatmap
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Final Dataset
print(df.head())

# Save Processed Dataset
df.to_csv("Processed_Titanic.csv", index=False)

print("Data Preprocessing Completed Successfully!")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Titanic dataset
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Data Preprocessing

# Handle missing values in Age, Embarked, and drop Cabin
train_df['Age'].fillna(train_df['Age'].median(), inplace=True)
train_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace=True)
train_df.drop(columns=['Cabin'], inplace=True)

# Encode 'Sex' column (Male=0, Female=1)
train_df['Sex'] = train_df['Sex'].map({'male': 0, 'female': 1})

# One-hot encode 'Embarked' (C, Q, S)
train_df = pd.get_dummies(train_df, columns=['Embarked'], drop_first=True)

# Select relevant features
X = train_df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_Q', 'Embarked_S']]
y = train_df['Survived']

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models

# 1. Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_val)

# 2. Logistic Regression
lr = LogisticRegression(max_iter=200)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_val)

# 3. Support Vector Classifier (SVC)
svc = SVC(kernel='linear')
svc.fit(X_train, y_train)
y_pred_svc = svc.predict(X_val)

# Evaluate models

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Plot confusion matrices for each model
plot_confusion_matrix(y_val, y_pred_rf, 'Random Forest')
plot_confusion_matrix(y_val, y_pred_lr, 'Logistic Regression')
plot_confusion_matrix(y_val, y_pred_svc, 'SVC')

# Print accuracy scores for each model
print("Random Forest Accuracy:", accuracy_score(y_val, y_pred_rf))
print("Logistic Regression Accuracy:", accuracy_score(y_val, y_pred_lr))
print("Support Vector Classifier Accuracy:", accuracy_score(y_val, y_pred_svc))

# Make predictions on the test dataset

# Preprocess test dataset similarly
test_df['Age'].fillna(test_df['Age'].median(), inplace=True)
test_df['Fare'].fillna(test_df['Fare'].median(), inplace=True)
test_df['Sex'] = test_df['Sex'].map({'male': 0, 'female': 1})
test_df = pd.get_dummies(test_df, columns=['Embarked'], drop_first=True)

# Select relevant features for the test dataset
X_test = test_df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_Q', 'Embarked_S']]

# Predict using the Random Forest model (you can replace it with any other trained model)
y_pred_test = rf.predict(X_test)

# Prepare submission file
submission = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Survived': y_pred_test})
submission.to_csv('submission.csv', index=False)

print("Prediction on test dataset completed. Submission file saved as 'submission.csv'.")

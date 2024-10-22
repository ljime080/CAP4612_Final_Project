# Import necessary packages for Logistic Regression
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

# Load the dataset
bear_attacks = pd.read_csv('C:/Users/monso/Desktop/bear_attacks.csv')

# Define a function to identify fatal attacks from the 'Details' column
def is_fatal(details):
    fatal_keywords = [
        'died', 'killed', 'fatal', 'dead', 'mauled to death', 'succumbed',
        'did not survive', 'passed away', 'pronounced dead', 'lost his life',
        'lost her life', 'slain', 'fatally injured', 'mortally wounded'
    ]
    return any(keyword in details.lower() for keyword in fatal_keywords)

# Create a new 'Fatal' column based on the function
bear_attacks['Fatal'] = bear_attacks['Details'].apply(is_fatal).astype(int)

# Convert 'Age' column to numeric, coercing errors
bear_attacks['Age'] = pd.to_numeric(bear_attacks['Age'], errors='coerce')

# Check if conversion was successful
print(bear_attacks['Age'].head())  # Check the first few rows of the 'Age' column

# Handle missing data in 'Age' column by filling NaN values with the mean age
bear_attacks['Age'].fillna(bear_attacks['Age'].mean(), inplace=True)

# Preprocess data: select relevant features and target
X = bear_attacks[['Latitude', 'Longitude', 'Age']]  # Features: latitude, longitude, age
y = bear_attacks['Fatal']  # Target: Fatal

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6, random_state=42)

# Optionally scale the features for better performance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train the Logistic Regression model
bear_model = LogisticRegression(random_state=42)
bear_model.fit(X_train, np.ravel(y_train))

# Predict on the test set
y_pred = bear_model.predict(X_test)

# Evaluate the model
print(f"Accuracy: {metrics.accuracy_score(y_test, y_pred)}")
print("Confusion Matrix:")
print(metrics.confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(metrics.classification_report(y_test, y_pred))

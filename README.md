# Heart-Failure-Prediction-using-machine-leraning
Machine learning project predicting heart failure based on medical features using Kaggle dataset. Includes data analysis, cleaning, feature selection, and model selection. Best model is random forest with 87% accuracy and important predictors are serum creatinine, age, and ejection fraction.

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the heart disease dataset
heart_df = pd.read_csv("C:\\Users\\hp\\Downloads\\archive (1)\\heart_failure_clinical_records_dataset.csv")

# Split the dataset into features and target variable
X = heart_df.drop("DEATH_EVENT", axis=1)
y = heart_df["DEATH_EVENT"]

# Split the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a logistic regression model
model = LogisticRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Evaluate the model's performance
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

Accuracy: 0.8
Confusion Matrix:
 [[33  2]
 [10 15]]

Project Description
The objective of this project was to utilize machine learning algorithms to predict heart failure based on medical features. We leveraged the heart failure clinical records dataset from Kaggle, which contains 13 features, including age, anaemia, diabetes, hypertension, smoking, and serum creatinine level, among others. The target variable is a binary indicator of whether the patient passed away due to heart failure during the follow-up period.

We began by performing an exploratory data analysis and data cleaning to prepare the dataset for modeling. We then utilized several machine learning algorithms such as logistic regression, k-nearest neighbors, decision trees, and random forests to predict heart failure. The models were assessed using multiple metrics, including accuracy, precision, recall, and F1 score. Additionally, we performed feature selection to identify the most significant predictors of heart failure.

Our findings indicate that the random forest algorithm was the most effective in predicting heart failure, yielding an accuracy of approximately 0.87 and an F1 score of 0.82. The most critical predictors of heart failure were serum creatinine level, age, and ejection fraction.

Project Conclusion
In conclusion, our study demonstrated the potential of machine learning algorithms for early detection and treatment of heart failure. We discovered that the random forest algorithm performed best in predicting heart failure, with an accuracy of around 0.87 and an F1 score of 0.82. Our analysis identified serum creatinine level, age, and ejection fraction as the most critical predictors of heart failure. These results can significantly contribute to improving patient outcomes and quality of life. Further research is required to improve the accuracy and reliability of heart failure prediction models, particularly in high-risk patient populations.

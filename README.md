# 🧠 Customer Churn Prediction with Logistic Regression

This project implements a customer churn prediction model using logistic regression. It utilizes customer data to predict whether a customer will churn or stay with the company. The notebook includes data preprocessing, feature encoding, model training, evaluation, and basic visualization.


 ## 🔍 Problem Statement

Customer churn is a critical metric for businesses in competitive markets. The goal of this project is to build a machine learning model that predicts whether a customer will churn based on their demographic, service usage, and account-related features.

---

## 🚀 Features

- Data cleaning and preprocessing
- Label and one-hot encoding of categorical features
- Logistic Regression model with `scikit-learn`
- Model evaluation using:
  - Confusion Matrix
  - Classification Report
  - Accuracy Score
- Feature importance visualization via bar chart

---

## 📦 Dependencies

To run this notebook, install the following Python libraries:

pip install pandas numpy matplotlib seaborn scikit-learn

## 📊 Model Used
Logistic Regression

Chosen for its interpretability and performance on binary classification tasks.
Random state fixed at 42 for reproducibility.

📈 Evaluation Metrics
Accuracy Score: Overall correctness of the model
Confusion Matrix: Breakdown of true/false positives/negatives
Classification Report: Precision, recall, F1-score per class

📌 Sample Code (Model Training & Evaluation)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train)

y_pred = log_reg.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred)


## 📊 Visualization
The notebook includes a horizontal bar chart of feature importances (coefficients) to help interpret which variables most influence churn prediction.

## 👩‍💻 Author
Vivian Phung
Email: phungvi08123@gmail.com

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 💡 Future Work
Add more ML models for comparison (e.g., Random Forest, XGBoost)
Use cross-validation and grid search for hyperparameter tuning
Deploy the model as an API

Let me know if you'd like it in `.md` format or want a version adapted for GitHub Pages.

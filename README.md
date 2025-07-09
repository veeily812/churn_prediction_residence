# 🧠 Customer Churn Prediction with Logistic Regression

This project implements a customer churn prediction model using different model to see which give the most appropriate result. It utilizes customer data to predict whether a customer will churn or stay with the company. The notebook includes data preprocessing, feature encoding, model training, evaluation, and basic visualization.


 ## 🔍 Problem Statement

Customer churn is a critical metric for businesses in competitive markets. The goal of this project is to build a machine learning model that predicts whether a customer will churn based on their demographic, service usage, and account-related features.

---

## 🚀 Features

- Data cleaning and preprocessing
- Label and one-hot encoding of categorical features
- RandomForestClassifier, Logistic Regression, SVM, KNearestNeighbor model with `scikit-learn`  -->  RandomForestClassifier is lowest with 77% accuracy, others: 80%
- Model evaluation using:
  - Confusion Matrix
  - Classification Report
  - Accuracy Score
- Feature importance visualization via bar chart
<img width="785" alt="Screenshot 2025-07-09 at 15 42 43" src="https://github.com/user-attachments/assets/3602b7af-a7f5-4ad2-8d1f-ea7e558b105c" />

---

## 📦 Dependencies

To run this notebook, install the following Python libraries:

pip install pandas numpy matplotlib seaborn scikit-learn

## 📊 Model Used
Logistic Regression, RandomForestClassifier, SVM, KNeighborsClassifier

Chosen for its interpretability and performance on binary classification tasks.
Random state fixed at 42 for reproducibility.

📈 Evaluation Metrics
Accuracy Score: Overall correctness of the model
Confusion Matrix: Breakdown of true/false positives/negatives
Classification Report: Precision, recall, F1-score per class


## 📊 Visualization
The notebook includes a horizontal bar chart of feature importances (coefficients) to help interpret which variables most influence churn prediction.
1. RandomForestClassifier
<img width="522" alt="Screenshot 2025-07-09 at 15 34 27" src="https://github.com/user-attachments/assets/9a12d0c3-fe35-496f-9bb4-b463d9d275d6" />

2. Logistic Regression
<img width="602" alt="Screenshot 2025-07-09 at 15 35 16" src="https://github.com/user-attachments/assets/fd7a3101-7a07-4020-9efb-089cd0a2c064" />

3. SVM
<img width="608" alt="Screenshot 2025-07-09 at 15 35 31" src="https://github.com/user-attachments/assets/7917e253-1750-4115-9964-0fdc3ae6aa2f" />

4. KNeighborsClassifier
<img width="582" alt="Screenshot 2025-07-09 at 15 36 09" src="https://github.com/user-attachments/assets/beaf22fb-38f9-47ee-959b-02c62e19c173" />


## 👩‍💻 Author
Vivian Phung
Email: phungvi08123@gmail.com

## 💡 Future Work
- Use cross-validation and grid search for hyperparameter tuning
- Deploy the model as an API

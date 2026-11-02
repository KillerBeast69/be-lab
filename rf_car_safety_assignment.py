"""
====================================================================
ASSIGNMENT: Car Safety/Evaluation Prediction using Random Forest
====================================================================

1. ASSIGNMENT DETAILS
---------------------
Objective: Implement a Random Forest Classifier model to predict the 
safety/acceptability of a car based on its attributes (buying price, 
maintenance, doors, persons, lug_boot, safety).
Dataset Link: https://www.kaggle.com/datasets/elikplim/car-evaluation-data-set

Note: In the standard UCI Car Evaluation dataset, all features are 
categorical. The target variable is usually the 'class' (acceptability: 
unacc, acc, good, vgood).

2. VIVA THEORY
--------------
- What is a Random Forest?
  It is an ensemble supervised machine learning algorithm used for both 
  classification and regression. It builds multiple decision trees and 
  merges their outputs to get a more accurate and stable prediction.
- What is Ensemble Learning (Bagging)?
  Ensemble learning combines multiple models. Random Forest uses "Bagging" 
  (Bootstrap Aggregating). It trains each tree on a random subset of data 
  and a random subset of features, reducing the variance/overfitting seen 
  in single decision trees.
- Why use Random Forest here?
  The car evaluation dataset consists entirely of categorical data. 
  Decision trees natively handle categorical/ordinal splits very well, 
  and Random Forest prevents the model from memorizing (overfitting) the data.
- What is Feature Importance?
  Random Forest can calculate how much each feature contributes to decreasing 
  impurity across all trees, allowing us to see which car attribute (like 
  'safety' or 'persons') matters most for the final evaluation.

3. CODE DETAILS
---------------
- pandas (pd): Loads the dataset. We explicitly define column names since 
  the raw CSV often lacks a header row.
- LabelEncoder: Converts categorical string labels (like 'vhigh', 'low', '2') 
  into machine-readable numerical values (like 3, 0, 1).
- train_test_split: Divides the dataset (typically 80% train, 20% test).
- RandomForestClassifier: The main model. 'n_estimators=100' means it builds 
  100 distinct decision trees.
- classification_report & confusion_matrix: Evaluates the model's accuracy, 
  precision, recall, and f1-score.
- Feature Importance Plot: Visualizes which car features had the highest 
  predictive power.

4. IMPLEMENTATION & EXECUTION
-----------------------------
Prerequisites (Necessary Packages):
- pandas, scikit-learn, matplotlib, seaborn

How to run:
1. Download 'car_evaluation.csv' from the Kaggle link and place it in the 
   same directory as this script.
2. Open your terminal or command prompt.
3. Install required packages:
   pip install pandas scikit-learn matplotlib seaborn
4. Save this code in a file named `rf_car_safety.py`.
5. Execute the script:
   python rf_car_safety.py
====================================================================
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os

def main():
    print("1. Loading the Car Evaluation dataset...")
    
    # Define column names as they are usually missing in the raw CSV
    col_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
    
    try:
        # Load dataset
        # In Kaggle, the file might be named 'car_evaluation.csv'
        df = pd.read_csv('car_evaluation.csv', names=col_names, header=None)
        
        # If the downloaded CSV already has a header, drop the first row if it matches col_names
        if df.iloc[0]['buying'] == 'buying':
            df = df[1:]
    except FileNotFoundError:
        print("ERROR: 'car_evaluation.csv' not found.")
        print("Please download it from Kaggle and place it in this folder.")
        return

    print(f"-> Dataset loaded successfully. Shape: {df.shape}")

    print("\n2. Pre-processing the dataset...")
    # Initialize LabelEncoder
    le = LabelEncoder()
    
    # Apply LabelEncoder to all columns (since all are categorical)
    df_encoded = pd.DataFrame()
    for col in df.columns:
        df_encoded[col] = le.fit_transform(df[col])
        
    print("-> Categorical features encoded into numerical values.")

    # Separate features (X) and target class (y)
    X = df_encoded.drop('class', axis=1)
    y = df_encoded['class']

    # Train-test split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\n3. Training the Random Forest Classifier...")
    # Initialize and train the model
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_classifier.fit(X_train, y_train)

    print("\n4. Evaluating the Model...")
    # Predict on test data
    y_pred = rf_classifier.predict(X_test)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Overall Accuracy: {accuracy * 100:.2f}%\n")
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    print("\n5. Generating Visualizations...")
    
    # 1. Plot Feature Importances
    importances = rf_classifier.feature_importances_
    feature_names = X.columns
    
    # Sort features by importance
    feature_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_imp_df = feature_imp_df.sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=(8, 5))
    sns.barplot(x='Importance', y='Feature', data=feature_imp_df, palette='viridis')
    plt.title('Random Forest - Feature Importance (Car Safety/Eval)')
    plt.tight_layout()
    plt.savefig('rf_feature_importance.png')
    print("-> Saved feature importance plot as 'rf_feature_importance.png'")

    # 2. Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix for Car Evaluation')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.tight_layout()
    plt.savefig('rf_confusion_matrix.png')
    print("-> Saved confusion matrix plot as 'rf_confusion_matrix.png'")
    
    print("\nExecution Complete. Please check the directory for generated image plots.")

if __name__ == "__main__":
    main()

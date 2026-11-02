"""
====================================================================
ASSIGNMENT: Data Cleaning and Preparation (Telecom Churn)
====================================================================

1. ASSIGNMENT DETAILS
---------------------
Objective: Perform end-to-end data cleaning, preprocessing, feature 
engineering, and dataset splitting on a Telecom Customer Churn dataset.
Note: Since no dataset link was provided, the script automatically 
generates a mock "Telecom_Customer_Churn.csv" with intentional errors 
(missing values, duplicates, outliers) to demonstrate all tasks perfectly.

2. VIVA THEORY
--------------
- What is Data Cleaning?
  The process of fixing errors, omissions, and inconsistencies in data 
  (e.g., removing duplicates, filling missing values, fixing typos).
- Why handle outliers?
  Outliers can heavily skew statistical models and machine learning 
  algorithms (especially linear regression and K-Means). We typically 
  cap them using the Interquartile Range (IQR) or Z-scores.
- What is Feature Engineering?
  Creating new, meaningful variables from existing ones to help machine 
  learning models find better patterns. (e.g., calculating 'Avg_Charge' 
  from 'TotalCharges' and 'Tenure').
- Why Normalize or Scale data?
  Features often have different scales (e.g., Tenure is 1-72, while 
  TotalCharges is 100-8000). Scaling ensures algorithms that compute 
  distances (like KNN, SVM) don't heavily weight the larger numbers.
- Why split the dataset into Train/Test?
  To evaluate how well our future model generalizes to unseen data. 
  Training data is for learning; testing data is for evaluation.

3. CODE DETAILS & TASKS COVERED
-------------------------------
- Task 1 & 2: pd.read_csv() and df.info() to load and explore.
- Task 3 & 4: df.fillna() handles NaNs, drop_duplicates() removes dupes.
- Task 5: df['gender'].replace() standardizes 'M'/'F' to 'Male'/'Female'.
- Task 6: pd.to_numeric() fixes string-encoded numbers.
- Task 7: IQR (Interquartile Range) method used to cap outliers.
- Task 8: New feature 'Avg_Charge_Per_Month' is created.
- Task 9: StandardScaler standardizes numerical columns.
- Task 10: train_test_split() divides the data (80% train, 20% test).
- Task 11: to_csv() exports the final cleaned dataset.

4. IMPLEMENTATION & EXECUTION
-----------------------------
Prerequisites: pandas, numpy, scikit-learn

How to run:
1. Open your terminal or command prompt.
2. Install required packages: pip install pandas numpy scikit-learn
3. Save this code in a file named `telecom_churn_cleaning.py`.
4. Execute the script: python telecom_churn_cleaning.py
====================================================================
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import warnings
warnings.filterwarnings('ignore')

def generate_mock_dataset():
    """Generates a dataset with intentional dirty data to demonstrate cleaning."""
    data = {
        'CustomerID': ['C01', 'C02', 'C03', 'C04', 'C05', 'C06', 'C07', 'C08', 'C09', 'C09'], # Contains duplicate C09
        'Gender': ['Male', 'F', 'Female', 'M', 'Female', 'Male', 'M', 'Female', 'Male', 'Male'], # Inconsistent
        'Tenure': [12, 45, 2, np.nan, 60, 24, 72, 5, 120, 120], # nan, and 120 is an outlier
        'MonthlyCharges': [50.5, 89.0, 20.0, 105.5, 110.0, 65.0, 25.0, 95.0, 200.0, 200.0], # 200 is an outlier
        'TotalCharges': ['606.0', '4005.0', '40.0', ' ', '6600.0', '1560.0', '1800.0', '475.0', '24000.0', '24000.0'], # String with empty space
        'Churn': ['No', 'No', 'Yes', 'Yes', 'No', 'No', 'No', 'Yes', 'Yes', 'Yes']
    }
    df = pd.DataFrame(data)
    df.to_csv('Telecom_Customer_Churn.csv', index=False)
    print("-> Mock 'Telecom_Customer_Churn.csv' created with dirty data.")

def main():
    print("0. Initializing Mock Dataset...")
    generate_mock_dataset()

    print("\nTask 1 & 2: Importing and Exploring Data...")
    df = pd.read_csv('Telecom_Customer_Churn.csv')
    print("Initial Shape:", df.shape)
    
    print("\nTask 4: Removing Duplicates...") 
    # Doing this before filling NaNs prevents double-counting issues
    df = df.drop_duplicates()
    print("Shape after dropping duplicates:", df.shape)

    print("\nTask 6: Fixing Data Types...")
    # 'TotalCharges' usually has an empty space string ' ' which fails normal float conversion. 
    # errors='coerce' forces it to NaN so we can handle it in the missing values step.
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    print("Converted TotalCharges to numeric. Missing TotalCharges count:", df['TotalCharges'].isna().sum())

    print("\nTask 3: Handling Missing Values...")
    # Fill missing Tenure with median, and missing TotalCharges with Tenure * MonthlyCharges
    df['Tenure'] = df['Tenure'].fillna(df['Tenure'].median())
    df['TotalCharges'] = df['TotalCharges'].fillna(df['Tenure'] * df['MonthlyCharges'])
    print("Missing values handled.")

    print("\nTask 5: Standardizing Inconsistent Data...")
    # Fix 'M' and 'F' to 'Male' and 'Female'
    df['Gender'] = df['Gender'].replace({'M': 'Male', 'F': 'Female'})
    print("Standardized Gender column:", df['Gender'].unique())

    print("\nTask 7: Handling Outliers (using IQR method)...")
    for col in ['Tenure', 'MonthlyCharges']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        upper_bound = Q3 + 1.5 * IQR
        lower_bound = Q1 - 1.5 * IQR
        
        # Cap outliers instead of dropping them to preserve data
        df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
        df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
    print("Outliers capped for numerical features.")

    print("\nTask 8: Feature Engineering...")
    # Create a new feature 'Avg_Charge_Per_Month'
    # Adding a small epsilon (1e-5) to avoid division by zero if Tenure is 0
    df['Avg_Charge_Per_Month'] = df['TotalCharges'] / (df['Tenure'] + 1e-5)
    print("New feature 'Avg_Charge_Per_Month' created.")

    print("\nTask 9: Scaling the Data...")
    # We only scale numerical columns
    scaler = StandardScaler()
    num_cols = ['Tenure', 'MonthlyCharges', 'TotalCharges', 'Avg_Charge_Per_Month']
    df[num_cols] = scaler.fit_transform(df[num_cols])
    print("Numerical features standardized (Mean=0, Std=1).")

    print("\nTask 11: Exporting Cleaned Dataset...")
    cleaned_file = 'Cleaned_Telecom_Customer_Churn.csv'
    df.to_csv(cleaned_file, index=False)
    print(f"Cleaned dataset saved as '{cleaned_file}'.")

    print("\nTask 10: Splitting the Dataset...")
    X = df.drop(['CustomerID', 'Churn'], axis=1) # Drop ID and Target
    y = df['Churn']
    
    # 80% Training, 20% Testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Dataset split completed. Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    print("\nExecution Complete! Cleaned data is ready for modeling.")

if __name__ == "__main__":
    main()

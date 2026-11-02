"""
====================================================================
ASSIGNMENT: Uber Fare Prediction using Regression Models
====================================================================

1. ASSIGNMENT DETAILS
---------------------
Objective: Predict the price of an Uber ride based on pickup and drop-off 
locations (and other features). The tasks include data pre-processing, 
identifying outliers, checking feature correlation, implementing Linear, 
Ridge, and Lasso regression models, and evaluating/comparing their performance 
using metrics like R2 and RMSE.
Dataset Note: You will need to download 'uber.csv' from Kaggle:
https://www.kaggle.com/datasets/yasserh/uber-fares-dataset

2. VIVA THEORY
--------------
- What is Regression? 
  A supervised machine learning technique used to predict continuous 
  numerical values (like price/fare).
- What is Linear Regression?
  It assumes a linear relationship between the input features (X) and the 
  target variable (y), finding the "line of best fit" that minimizes the 
  sum of squared residuals.
- What are Ridge and Lasso Regression?
  They are regularized versions of linear regression designed to prevent 
  overfitting by adding a penalty term to the loss function.
  * Ridge (L2 regularization): Penalizes the square of the coefficients.
  * Lasso (L1 regularization): Penalizes the absolute value of coefficients, 
    often driving some to exactly zero (performing feature selection).
- Why identify outliers?
  Regression models (especially linear ones using Mean Squared Error) are 
  highly sensitive to outliers. Extreme values can skew the line of best fit.
- What are R2 and RMSE?
  * R-squared (R2): The proportion of variance in the target variable that 
    can be explained by the model (higher is better, max 1.0).
  * Root Mean Squared Error (RMSE): The standard deviation of the residuals 
    (prediction errors). It tells you how concentrated the data is around 
    the line of best fit (lower is better).

3. CODE DETAILS
---------------
- Pre-processing: We drop unnecessary columns (like IDs), handle missing 
  values, convert dates to useful numerical features (hour, day, etc.), and 
  calculate the Haversine distance (actual distance in km between coordinates).
- Outliers: We use the IQR (Interquartile Range) method to cap extreme 
  values in fare_amount and distance to prevent them from skewing the models.
- Correlation: We compute a correlation matrix and plot a heatmap to see 
  which features strongly relate to the fare amount (distance should be high).
- Modeling: We split data into training and testing sets, standardize the 
  features, and train LinearRegression, Ridge, and Lasso models.
- Evaluation: We calculate R2, RMSE, and MAE for each model and print a 
  comparison.

4. IMPLEMENTATION & EXECUTION
-----------------------------
Prerequisites (Necessary Packages):
- pandas, numpy, scikit-learn, matplotlib, seaborn, math

How to run:
1. Download the dataset from Kaggle and save it as 'uber.csv' in the same 
   directory as this script.
2. Open your terminal or command prompt.
3. Install required packages:
   pip install pandas numpy scikit-learn matplotlib seaborn
4. Save this code in a file named `uber_fare_prediction.py`.
5. Execute the script:
   python uber_fare_prediction.py
====================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Function to calculate distance between coordinates
def haversine(lon1, lat1, lon2, lat2):
    """Calculates the great circle distance between two points on the earth."""
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371 # Radius of earth in kilometers
    return c * r

def main():
    print("1. Loading and Pre-processing Dataset...")
    try:
        # Load dataset (Make sure uber.csv is in the directory)
        df = pd.read_csv('uber.csv')
    except FileNotFoundError:
        print("ERROR: 'uber.csv' not found. Please download it from Kaggle and place it in this folder.")
        return

    # Drop irrelevant columns
    if 'Unnamed: 0' in df.columns:
        df.drop('Unnamed: 0', axis=1, inplace=True)
    if 'key' in df.columns:
        df.drop('key', axis=1, inplace=True)

    # Drop rows with missing values
    df.dropna(inplace=True)

    # Convert datetime and extract features
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    df['hour'] = df['pickup_datetime'].dt.hour
    df['day'] = df['pickup_datetime'].dt.day
    df['month'] = df['pickup_datetime'].dt.month
    df['dayofweek'] = df['pickup_datetime'].dt.dayofweek
    df.drop('pickup_datetime', axis=1, inplace=True)

    # Filter out illogical coordinate anomalies (e.g., latitude > 90 or < -90)
    df = df[(df['pickup_latitude'] >= -90) & (df['pickup_latitude'] <= 90)]
    df = df[(df['dropoff_latitude'] >= -90) & (df['dropoff_latitude'] <= 90)]
    df = df[(df['pickup_longitude'] >= -180) & (df['pickup_longitude'] <= 180)]
    df = df[(df['dropoff_longitude'] >= -180) & (df['dropoff_longitude'] <= 180)]
    
    # Calculate distance in km
    df['distance_km'] = df.apply(lambda row: haversine(row['pickup_longitude'], row['pickup_latitude'], 
                                                       row['dropoff_longitude'], row['dropoff_latitude']), axis=1)

    print("\n2. Identifying and Handling Outliers...")
    # Remove rows where fare is less than or equal to 0 or distance is 0 but fare is high
    df = df[df['fare_amount'] > 0]
    
    # Cap outliers using IQR method for fare and distance
    for col in ['fare_amount', 'distance_km']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # Instead of dropping, we cap them to preserve data while reducing skew
        df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
        df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])

    print("\n3. Checking Correlation...")
    # Plot correlation heatmap
    plt.figure(figsize=(10, 8))
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
    print("Saved correlation heatmap as 'correlation_heatmap.png'")

    print("\n4. Implementing Regression Models...")
    # Prepare data for modeling
    X = df.drop('fare_amount', axis=1)
    y = df['fare_amount']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize models
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.1)
    }

    print("\n5. Evaluating and Comparing Models...")
    results = {}
    for name, model in models.items():
        # Train
        model.fit(X_train_scaled, y_train)
        # Predict
        y_pred = model.predict(X_test_scaled)
        
        # Evaluate
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        results[name] = {'R2': r2, 'RMSE': rmse, 'MAE': mae}
        
    # Print comparison
    print("-" * 50)
    print(f"{'Model':<20} | {'R2 Score':<10} | {'RMSE':<10}")
    print("-" * 50)
    for name, metrics in results.items():
        print(f"{name:<20} | {metrics['R2']:.4f}     | {metrics['RMSE']:.4f}")
    print("-" * 50)
    
    print("\nExecution Complete. Please check the directory for generated plots.")

if __name__ == "__main__":
    main()

"""
====================================================================
ASSIGNMENT: Data Wrangling on Real Estate Market
====================================================================

1. ASSIGNMENT DETAILS
---------------------
Objective: Perform data wrangling on housing prices data to prepare it 
for analysis. This includes cleaning column names, handling missing 
values, merging datasets, filtering, encoding, aggregating, and 
handling outliers.
Note: Since no dataset link was provided, a mock main dataset 
('RealEstate_Prices.csv') and a secondary dataset ('Neighborhood_Info.csv') 
are generated on the fly to perfectly demonstrate the merge task.

2. VIVA THEORY
--------------
- What is Data Wrangling?
  Also known as data munging, it is the process of transforming and 
  mapping raw data into another format to make it more appropriate 
  and valuable for analytics and machine learning.
- Why clean column names?
  Spaces and special characters (like '$', '-', or '!') in column headers 
  cause syntax errors in pandas and SQL. Replacing them with underscores 
  ensures smooth programmatic access (e.g., df.Sale_Price).
- What is the difference between One-Hot Encoding and Label Encoding?
  * One-Hot Encoding creates a new binary column for each category 
    (good for nominal data without inherent order, like Property Type).
  * Label Encoding assigns an integer to each category (good for 
    ordinal data, like 'Low', 'Medium', 'High').
- Why perform Data Merging?
  To enrich the primary dataset. Housing prices are heavily influenced 
  by external factors, so joining a neighborhood demographics table 
  provides better predictive features.

3. CODE DETAILS & TASKS COVERED
-------------------------------
- Task 1: `str.replace()` cleans messy column names via Regex.
- Task 2: `fillna(median)` imputes missing numeric values safely.
- Task 3: `pd.merge()` performs a SQL-like left join with a secondary 
  neighborhood dataset.
- Task 7: IQR capping is applied to 'Square_Footage' and 'Sale_Price' 
  to neutralize extreme mansion prices skewing the data.
- Task 4: `df[df[...].isin()]` filters the data to specific property types.
- Task 6: `groupby().mean()` calculates average prices by neighborhood.
- Task 5: `pd.get_dummies()` performs One-Hot Encoding on categories.

4. IMPLEMENTATION & EXECUTION
-----------------------------
Prerequisites: pandas, numpy

How to run:
1. Open your terminal or command prompt.
2. Install required packages: pip install pandas numpy
3. Save this code in a file named `data_wrangling_realestate.py`.
4. Execute the script: python data_wrangling_realestate.py
====================================================================
"""

import pandas as pd
import numpy as np
import re

def generate_mock_datasets():
    """Generates two dirty datasets to demonstrate wrangling and merging."""
    # Main dataset with spaces and special chars in columns, plus NaNs and outliers
    main_data = {
        'Property ID': [1, 2, 3, 4, 5, 6, 7],
        'Neighborhood Area': ['Downtown', 'Suburb', 'Downtown', 'Uptown', 'Suburb', 'Uptown', 'Suburb'],
        'Property-Type!': ['House', 'Condo', 'House', 'Apartment', 'House', 'House', 'Condo'],
        'Square Footage': [1500, np.nan, 1200, 800, 2500, 15000, 1100], # 15000 is an extreme outlier, NaN included
        'Sale Price ($)': [300000, 200000, 250000, 150000, 400000, 5000000, 210000] # 5,000,000 is an outlier
    }
    # Secondary dataset for merging
    extra_data = {
        'Neighborhood Area': ['Downtown', 'Suburb', 'Uptown'],
        'Crime Rate Index': ['High', 'Low', 'Medium'],
        'Schools Rating': [7, 9, 8]
    }
    pd.DataFrame(main_data).to_csv('RealEstate_Prices.csv', index=False)
    pd.DataFrame(extra_data).to_csv('Neighborhood_Info.csv', index=False)
    print("0. Mock datasets 'RealEstate_Prices.csv' and 'Neighborhood_Info.csv' generated.")

def main():
    generate_mock_datasets()
    
    print("\nTask 1: Importing Data & Cleaning Column Names...")
    df = pd.read_csv('RealEstate_Prices.csv')
    # Replace non-alphanumeric characters with underscores and strip trailing/leading underscores
    df.columns = df.columns.str.replace(r'[^a-zA-Z0-9]', '_', regex=True).str.strip('_')
    # Consolidate multiple underscores into a single one
    df.columns = df.columns.str.replace(r'_+', '_', regex=True)
    print("-> Cleaned Columns:", df.columns.tolist())
    
    print("\nTask 2: Handling Missing Values...")
    # Impute missing Square Footage with the median of the column
    missing_before = df['Square_Footage'].isna().sum()
    df['Square_Footage'] = df['Square_Footage'].fillna(df['Square_Footage'].median())
    print(f"-> Imputed {missing_before} missing value(s) in 'Square_Footage' with median.")

    print("\nTask 3: Merging Additional Datasets...")
    df_extra = pd.read_csv('Neighborhood_Info.csv')
    df_extra.columns = df_extra.columns.str.replace(r'[^a-zA-Z0-9]', '_', regex=True).str.strip('_')
    # Left join on the Neighborhood column
    df_merged = pd.merge(df, df_extra, on='Neighborhood_Area', how='left')
    print("-> Datasets merged successfully. New columns added:", df_extra.columns.drop('Neighborhood_Area').tolist())

    print("\nTask 7: Handling Outliers (Extreme Values)...")
    # Identify and cap outliers in Square Footage and Sale Price using IQR
    for col in ['Square_Footage', 'Sale_Price_']:
        Q1 = df_merged[col].quantile(0.25)
        Q3 = df_merged[col].quantile(0.75)
        IQR = Q3 - Q1
        upper_bound = Q3 + 1.5 * IQR
        # Cap extreme values to the upper bound
        df_merged[col] = np.where(df_merged[col] > upper_bound, upper_bound, df_merged[col])
    print("-> Outliers capped using the IQR method.")

    print("\nTask 4: Filtering and Subsetting Data...")
    # Filter to only include Houses and Condos (exclude Apartments for this specific analysis)
    df_filtered = df_merged[df_merged['Property_Type_'].isin(['House', 'Condo'])]
    print(f"-> Filtered out Apartments. Remaining records: {len(df_filtered)}")

    print("\nTask 6: Aggregating Data (Summary Statistics)...")
    # Calculate average sale prices by neighborhood
    avg_price_by_hood = df_filtered.groupby('Neighborhood_Area')['Sale_Price_'].mean().round(2)
    print("-> Average Sale Price by Neighborhood:")
    print(avg_price_by_hood)

    print("\nTask 5: Encoding Categorical Variables...")
    # One-Hot Encoding for 'Property_Type' and 'Crime_Rate_Index'
    # drop_first=True prevents the dummy variable trap (multicollinearity)
    df_final = pd.get_dummies(df_filtered, columns=['Property_Type_', 'Crime_Rate_Index'], drop_first=True)
    print("-> One-Hot Encoding applied. Final dataset columns:")
    print(df_final.columns.tolist())

    # Export the final wrangled dataset
    output_file = 'Wrangled_RealEstate_Data.csv'
    df_final.to_csv(output_file, index=False)
    print(f"\nExecution Complete! Wrangled dataset exported to '{output_file}'.")

if __name__ == "__main__":
    main()

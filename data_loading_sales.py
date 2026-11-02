"""
====================================================================
ASSIGNMENT: Data Loading, Storage, and File Formats (Sales Data)
====================================================================

1. ASSIGNMENT DETAILS
---------------------
Objective: Load, clean, transform, analyze, and visualize sales data 
from multiple file formats (CSV, Excel, JSON). 
Syllabus: Fourth Year AI & Data Science (2020 Course), SPPU.
Note: Since no specific dataset link was provided, this script 
automatically generates 3 mock files (CSV, Excel, JSON) on the fly 
so you can execute it perfectly during your practical exam.

2. VIVA THEORY
--------------
- Why do we have different file formats?
  Different systems export data differently. CSV is lightweight and 
  universal, Excel supports multiple sheets and formatting, and JSON 
  is the standard for web APIs and NoSQL databases.
- What is Data Cleaning?
  The process of fixing or removing incorrect, corrupted, incorrectly 
  formatted, duplicate, or incomplete data (e.g., handling missing 
  values with dropna() or fillna()).
- What is Data Transformation?
  Converting data from one format or structure into another. In this 
  assignment, we merge three datasets into a unified structure and 
  derive a new variable (Total Sales = Quantity * Price).
- Why use Pandas for this?
  Pandas provides built-in, highly optimized I/O functions (read_csv, 
  read_excel, read_json) that automatically parse these formats into 
  a standardized DataFrame object.

3. CODE DETAILS
---------------
- create_dummy_data(): A helper function to create sample files so 
  the script works immediately without external downloads.
- pd.read_*: Loads the respective file formats.
- pd.concat(): Appends the three dataframes together into one unified 
  dataframe (Unified Format).
- dropna() & drop_duplicates(): Cleans the data.
- df['Total_Sales']: Derives a new column (Transformation).
- groupby(): Aggregates data by Product Category to find total sales.
- plt.subplots(): Creates visualizations (Bar and Pie charts) saved 
  as image files.

4. IMPLEMENTATION & EXECUTION
-----------------------------
Prerequisites (Necessary Packages):
- pandas, matplotlib, openpyxl (required for reading/writing Excel)

How to run:
1. Open your terminal or command prompt.
2. Install the required packages by running:
   pip install pandas matplotlib openpyxl
3. Save this code in a file named `data_loading_sales.py`.
4. Execute the script by running:
   python data_loading_sales.py
====================================================================
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

def create_dummy_data():
    """Generates mock CSV, Excel, and JSON files for the assignment."""
    data1 = {'OrderID': [1, 2, 3, 3], 'Product': ['Laptop', 'Mouse', 'Keyboard', 'Keyboard'], 'Qty': [1, 2, 1, 1], 'Price': [1000, 20, 50, 50]}
    data2 = {'OrderID': [4, 5, 6], 'Product': ['Monitor', 'Laptop', None], 'Qty': [2, 1, 1], 'Price': [200, 1000, 15]}
    data3 = [{'OrderID': 7, 'Product': 'Mouse', 'Qty': 5, 'Price': 20}, {'OrderID': 8, 'Product': 'Monitor', 'Qty': 1, 'Price': 200}]
    
    pd.DataFrame(data1).to_csv('sales.csv', index=False)
    pd.DataFrame(data2).to_excel('sales.xlsx', index=False)
    pd.read_json(str(data3).replace("'", '"'), orient='records').to_json('sales.json', orient='records')

def main():
    print("0. Generating mock data files (sales.csv, sales.xlsx, sales.json)...")
    create_dummy_data()

    # 1. Load the sales data from each file format
    print("1. Loading data from CSV, Excel, and JSON...")
    df_csv = pd.read_csv('sales.csv')
    df_excel = pd.read_excel('sales.xlsx')
    df_json = pd.read_json('sales.json')

    # 4. Convert to a unified format (Merge DataFrames)
    print("2 & 4. Merging into a unified DataFrame...")
    df_unified = pd.concat([df_csv, df_excel, df_json], ignore_index=True)
    
    # 2 & 3. Explore and Clean (Inconsistencies, missing values, duplicates)
    print(f"-> Original shape: {df_unified.shape}")
    df_clean = df_unified.drop_duplicates()  # Remove duplicates
    df_clean = df_clean.dropna()             # Remove missing values
    print(f"-> Cleaned shape (after dropping NaNs and duplicates): {df_clean.shape}")

    # 5. Data transformation (Deriving new variables)
    print("5. Deriving new variable: Total_Sales = Qty * Price...")
    df_clean['Total_Sales'] = df_clean['Qty'] * df_clean['Price']

    # 6. Analyze the sales data
    print("\n6. Analysis & Descriptive Statistics:")
    print("-" * 40)
    print(df_clean.describe())
    
    # Aggregation
    category_sales = df_clean.groupby('Product')['Total_Sales'].sum()
    avg_order_value = df_clean['Total_Sales'].mean()
    print("\nTotal Sales by Product Category:")
    print(category_sales)
    print(f"\nAverage Order Value: ${avg_order_value:.2f}")

    # 7. Create visualizations
    print("\n7. Generating Visualizations...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Bar Plot
    category_sales.plot(kind='bar', ax=axes[0], color='skyblue', edgecolor='black')
    axes[0].set_title('Total Sales by Product')
    axes[0].set_ylabel('Total Sales ($)')
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)

    # Pie Chart
    category_sales.plot(kind='pie', ax=axes[1], autopct='%1.1f%%', startangle=90, cmap='Pastel1')
    axes[1].set_title('Product Category Distribution')
    axes[1].set_ylabel('') # Hide y-label for pie chart

    plt.tight_layout()
    plt.savefig('sales_analysis_visuals.png')
    print("-> Saved visualizations as 'sales_analysis_visuals.png'")
    
    # Cleanup mock files (optional, left out to allow inspection)
    print("\nExecution Complete!")

if __name__ == "__main__":
    main()

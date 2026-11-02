"""
====================================================================
ASSIGNMENT: Data Aggregation (Retail Sales Performance)
====================================================================

1. ASSIGNMENT DETAILS
---------------------
Objective: Perform data aggregation to analyze sales performance by 
region and product category using a retail sales dataset.
Note: Since no dataset link was provided, the script automatically 
generates a mock "Retail_Sales_Data.csv" so all tasks can be executed 
and demonstrated perfectly during your exam.

2. VIVA THEORY
--------------
- What is Data Aggregation?
  The process of gathering data and presenting it in a summarized format. 
  It involves grouping multiple rows of data together to calculate 
  metrics like sum, average, minimum, or maximum.
- How does `groupby` work in Pandas?
  It follows a "Split-Apply-Combine" strategy:
  1. Split the data into groups based on some criteria (e.g., Region).
  2. Apply a function to each group independently (e.g., sum).
  3. Combine the results into a new data structure.
- What is a Stacked Bar Plot?
  A chart that uses bars to show comparisons between categories, with 
  each bar divided into sub-parts to show the cumulative effect of a 
  secondary categorical variable (like Product Category within a Region).
- What does `.unstack()` do?
  When you group by two columns (e.g., Region and Category), Pandas 
  creates a MultiIndex. `.unstack()` pivots the inner level (Category) 
  into columns, making it very easy to plot grouped or stacked charts.

3. CODE DETAILS & TASKS COVERED
-------------------------------
- Task 1 & 2: `pd.read_csv` loads data; `df.info()` explores it.
- Task 3: Relevant variables (Region, Product Category, Sales Amount) identified.
- Task 4: `groupby('Region')['Sales_Amount'].sum()` calculates total sales.
- Task 6: `sort_values(ascending=False)` identifies top-performing regions.
- Task 7: `groupby(['Region', 'Product_Category'])['Sales_Amount'].sum()` calculates combined sales.
- Task 5: `plt.pie()` visualizes the regional sales distribution.
- Task 8: `df.plot(kind='bar', stacked=True)` creates a stacked bar plot.

4. IMPLEMENTATION & EXECUTION
-----------------------------
Prerequisites: pandas, numpy, matplotlib

How to run:
1. Open your terminal or command prompt.
2. Install required packages: pip install pandas numpy matplotlib
3. Save this code in a file named `data_aggregation_sales.py`.
4. Execute the script: python data_aggregation_sales.py
====================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def generate_mock_dataset():
    """Generates a realistic Retail Sales dataset to demonstrate aggregation."""
    np.random.seed(42)
    regions = ['North', 'South', 'East', 'West', 'Central']
    categories = ['Electronics', 'Clothing', 'Home & Garden', 'Sports']
    
    n_records = 500
    df = pd.DataFrame({
        'Transaction_ID': range(1001, 1001 + n_records),
        'Transaction_Date': pd.date_range(start='2023-01-01', periods=n_records, freq='H'),
        'Region': np.random.choice(regions, n_records),
        'Product_Category': np.random.choice(categories, n_records),
        'Quantity': np.random.randint(1, 10, n_records),
        'Sales_Amount': np.random.uniform(50.0, 1500.0, n_records).round(2)
    })
    df.to_csv('Retail_Sales_Data.csv', index=False)
    print("0. Mock 'Retail_Sales_Data.csv' generated.")

def main():
    generate_mock_dataset()

    print("\nTask 1 & 2: Importing and Exploring Data...")
    df = pd.read_csv('Retail_Sales_Data.csv')
    print("Dataset Info:")
    print(df.info())
    
    print("\nTask 3: Identifying Relevant Variables...")
    print("-> Variables identified for analysis: 'Region', 'Product_Category', 'Sales_Amount'")
    
    print("\nTask 4: Grouping by Region (Total Sales)...")
    region_sales = df.groupby('Region')['Sales_Amount'].sum().round(2)
    print(region_sales)
    
    print("\nTask 6: Identifying Top-Performing Regions...")
    top_regions = region_sales.sort_values(ascending=False)
    print("-> Top Regions based on Sales Amount:")
    print(top_regions)
    print(f"** The overall top-performing region is {top_regions.index[0]} with ${top_regions.iloc[0]} in sales. **")
    
    print("\nTask 7: Grouping by Region and Product Category...")
    # Calculate total sales for each combination and unstack for easy plotting
    region_category_sales = df.groupby(['Region', 'Product_Category'])['Sales_Amount'].sum().unstack().round(2)
    print(region_category_sales)
    
    print("\nTask 5 & 8: Generating Visualizations...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Retail Sales Performance Dashboard', fontsize=16, fontweight='bold')
    
    # Task 5: Pie chart for sales distribution by region
    axes[0].pie(region_sales, labels=region_sales.index, autopct='%1.1f%%', 
                startangle=140, cmap='Set2', wedgeprops={'edgecolor': 'black'})
    axes[0].set_title('Total Sales Distribution by Region')
    
    # Task 8: Stacked bar plot for sales across regions and categories
    region_category_sales.plot(kind='bar', stacked=True, ax=axes[1], cmap='Pastel1', edgecolor='black')
    axes[1].set_title('Sales Amount by Region & Product Category')
    axes[1].set_ylabel('Total Sales Amount ($)')
    axes[1].set_xlabel('Region')
    axes[1].tick_params(axis='x', rotation=0)
    axes[1].legend(title='Product Category')
    axes[1].grid(axis='y', linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    output_filename = 'sales_aggregation_dashboard.png'
    plt.savefig(output_filename, dpi=300)
    print(f"-> Saved visual dashboard as '{output_filename}'")
    print("\nExecution Complete!")

if __name__ == "__main__":
    main()

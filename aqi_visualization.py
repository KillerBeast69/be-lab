"""
====================================================================
ASSIGNMENT: Data Visualization using Matplotlib (AQI Trends)
====================================================================

1. ASSIGNMENT DETAILS
---------------------
Objective: Use matplotlib to visualize Air Quality Index (AQI) trends 
and pollutant levels over time using Line, Bar, Box, and Scatter plots.
Note: Since no dataset link was provided, the script generates a mock 
"City_Air_Quality.csv" dataset so all tasks can be executed and 
demonstrated perfectly during your exam.

2. VIVA THEORY
--------------
- What is Matplotlib?
  It is a comprehensive library for creating static, animated, and 
  interactive visualizations in Python. The `pyplot` module provides a 
  MATLAB-like interface.
- Line Plots vs. Bar Plots:
  * Line plots are best for showing continuous data over time (trends).
  * Bar plots are best for comparing categorical data or discrete 
    aggregations (e.g., average AQI per month).
- What is a Box Plot?
  A standardized way of displaying the distribution of data based on 
  a five-number summary (minimum, first quartile (Q1), median, third 
  quartile (Q3), and maximum). It easily identifies outliers.
- What is a Scatter Plot?
  Used to observe relationships (correlations) between two continuous 
  variables (e.g., does AQI increase strictly as PM2.5 increases?).
- Why Subplots?
  `plt.subplots()` allows us to create multiple distinct charts within 
  a single figure, creating a comprehensive "dashboard" view.

3. CODE DETAILS & TASKS COVERED
-------------------------------
- Task 1 & 2: `pd.read_csv` loads data; `df.info()` explores it.
- Task 3: Relevant variables (Date, PM2.5, PM10, CO, AQI) are identified.
- Task 4: `ax.plot()` shows the overall AQI trend.
- Task 5: `ax.plot()` for individual pollutants (PM2.5, PM10, CO).
- Task 6: `ax.bar()` shows average AQI per month.
- Task 7: `ax.boxplot()` analyzes AQI distribution by pollutant category.
- Task 8: `ax.scatter()` explores AQI vs. PM2.5 relationships.
- Task 9: Customizations (titles, labels, legends, colors) are applied 
  globally across all subplots.

4. IMPLEMENTATION & EXECUTION
-----------------------------
Prerequisites: pandas, numpy, matplotlib

How to run:
1. Open your terminal or command prompt.
2. Install required packages: pip install pandas numpy matplotlib
3. Save this code in a file named `aqi_visualization.py`.
4. Execute the script: python aqi_visualization.py
====================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def generate_mock_dataset():
    """Generates realistic AQI data to demonstrate the visualizations."""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    pm25 = np.random.uniform(15, 120, 100)
    pm10 = pm25 * 1.5 + np.random.normal(10, 15, 100)
    co = np.random.uniform(0.5, 4.0, 100)
    
    # AQI is a derived metric. We'll simulate it as highly dependent on PM2.5
    aqi = pm25 * 1.2 + co * 10 + np.random.normal(0, 10, 100)
    
    # Dominant Pollutant category for the Box Plot
    dominant_pollutant = np.where(pm25 > 60, 'PM2.5', np.where(co > 2.5, 'CO', 'PM10'))
    
    df = pd.DataFrame({
        'Date': dates,
        'PM2.5': pm25,
        'PM10': pm10,
        'CO': co,
        'AQI': aqi,
        'Dominant_Pollutant': dominant_pollutant
    })
    df.to_csv('City_Air_Quality.csv', index=False)
    print("0. Mock 'City_Air_Quality.csv' generated.")

def main():
    generate_mock_dataset()

    print("\nTask 1, 2 & 3: Importing and Exploring Data...")
    df = pd.read_csv('City_Air_Quality.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.strftime('%Y-%m') # Extract month for aggregation
    
    print("Dataset Info:")
    print(df.info())
    print("\nFirst 3 rows:")
    print(df.head(3))

    print("\nTask 4 to 9: Generating Matplotlib Visualizations...")
    
    # Create a 3x2 grid of subplots for a dashboard-style output
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle('City Air Quality (AQI) Analysis Dashboard', fontsize=20, fontweight='bold')
    
    # Task 4: Line plot - Overall AQI trend over time
    ax = axes[0, 0]
    ax.plot(df['Date'], df['AQI'], color='crimson', linewidth=2, marker='o', markersize=4)
    ax.set_title('Overall AQI Trend Over Time')
    ax.set_ylabel('Air Quality Index (AQI)')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d')) # Format dates nicely
    ax.grid(True, linestyle='--', alpha=0.6)

    # Task 5: Line plots - Individual pollutant levels over time
    ax = axes[0, 1]
    ax.plot(df['Date'], df['PM2.5'], label='PM 2.5', color='blue', alpha=0.7)
    ax.plot(df['Date'], df['PM10'], label='PM 10', color='green', alpha=0.7)
    ax.set_title('Individual Pollutant Levels Over Time')
    ax.set_ylabel('Concentration (µg/m³)')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.6)

    # Task 6: Bar plot - Average AQI by Month
    ax = axes[1, 0]
    monthly_aqi = df.groupby('Month')['AQI'].mean()
    ax.bar(monthly_aqi.index, monthly_aqi.values, color='dodgerblue', edgecolor='black')
    ax.set_title('Average AQI per Month')
    ax.set_ylabel('Average AQI')
    ax.set_xlabel('Month')
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    # Task 7: Box plot - AQI Distribution by Dominant Pollutant
    ax = axes[1, 1]
    # Group AQI values into lists based on Dominant Pollutant
    categories = df['Dominant_Pollutant'].unique()
    aqi_data = [df[df['Dominant_Pollutant'] == cat]['AQI'] for cat in categories]
    
    box = ax.boxplot(aqi_data, patch_artist=True, labels=categories)
    # Adding colors to boxplot
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        
    ax.set_title('AQI Distribution by Dominant Pollutant')
    ax.set_ylabel('AQI Values')
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    # Task 8: Scatter plot - Relationship between AQI and PM2.5
    ax = axes[2, 0]
    scatter = ax.scatter(df['PM2.5'], df['AQI'], c=df['CO'], cmap='viridis', alpha=0.8, edgecolors='k', s=60)
    ax.set_title('AQI vs. PM 2.5 (Color mapped to CO levels)')
    ax.set_xlabel('PM 2.5 Concentration')
    ax.set_ylabel('Air Quality Index (AQI)')
    fig.colorbar(scatter, ax=ax, label='CO Level')
    ax.grid(True, linestyle='--', alpha=0.6)

    # Clean up empty subplot (since we have 5 charts for a 3x2 grid)
    fig.delaxes(axes[2, 1])

    # Task 9: Formatting & Customizations
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to not overlap title
    
    # Save the output
    output_filename = 'aqi_matplotlib_dashboard.png'
    plt.savefig(output_filename, dpi=300)
    print(f"-> Saved all visualizations as '{output_filename}'")
    print("\nExecution Complete!")

if __name__ == "__main__":
    main()

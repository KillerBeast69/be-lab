"""
====================================================================
ASSIGNMENT: Interacting with Web APIs (OpenWeatherMap)
====================================================================

1. ASSIGNMENT DETAILS
---------------------
Objective: Retrieve weather data from the OpenWeatherMap API, clean it, 
model patterns, aggregate statistics, and visualize trends, geographical 
data, and correlations.
Note: A fallback mock dataset is included so this script will execute 
perfectly during your exam even if you don't have a valid API key yet.

2. VIVA THEORY
--------------
- What is a Web API?
  Application Programming Interface. It allows different software systems 
  to communicate over the web, typically using HTTP requests and returning 
  data in JSON format.
- What is JSON?
  JavaScript Object Notation. A lightweight data-interchange format that 
  is easy for humans to read/write and machines to parse/generate.
- Why preprocess/clean API data?
  APIs can return deeply nested JSONs, missing fields (like missing 
  precipitation if it didn't rain), or inconsistent timestamps. We must 
  flatten and clean this for pandas to analyze it.
- What is a Correlation Heatmap?
  A graphical representation of the correlation matrix representing the 
  linear relationship between variables (e.g., if Humidity goes up, does 
  Temperature go down?). Values range from -1 to 1.

3. CODE DETAILS & TASKS COVERED
-------------------------------
- Task 1 & 2: `requests.get()` interacts with OpenWeatherMap.
- Task 3: JSON parsing extracts temp, humidity, wind, and rain.
- Task 4: `fillna(0)` cleans missing rain data; `pd.to_datetime` fixes dates.
- Task 5: `df.describe()` and direct math functions model min/max/avg.
- Task 6: `plt.plot` and `plt.bar` visualize temp and rain.
- Task 7: `df.groupby` aggregates stats by day.
- Task 8: `plt.scatter` using Longitude and Latitude maps the cities.
- Task 9: `sns.heatmap` visualizes attribute relationships.

4. IMPLEMENTATION & EXECUTION
-----------------------------
Prerequisites (Necessary Packages):
- requests, pandas, matplotlib, seaborn

How to run:
1. Open your terminal or command prompt.
2. Install required packages:
   pip install requests pandas matplotlib seaborn
3. (Optional) Replace 'YOUR_API_KEY' with a real OpenWeatherMap key.
4. Execute the script:
   python weather_api_assignment.py
====================================================================
"""

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Task 1: API Key setup
API_KEY = "YOUR_API_KEY_HERE"  # Replace with actual key if available
CITIES = ["Pune", "Mumbai", "Delhi", "Bangalore", "Chennai"]

def fetch_weather_data():
    """Task 2 & 3: Interact with API and extract relevant attributes."""
    weather_list = []
    
    print("Attempting to fetch data from OpenWeatherMap API...")
    for city in CITIES:
        # Using the 5-day / 3-hour forecast API
        url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={API_KEY}&units=metric"
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            for item in data['list']:
                weather_list.append({
                    'City': city,
                    'Lat': data['city']['coord']['lat'],
                    'Lon': data['city']['coord']['lon'],
                    'DateTime': item['dt_txt'],
                    'Temp': item['main']['temp'],
                    'Humidity': item['main']['humidity'],
                    'WindSpeed': item['wind']['speed'],
                    'Precipitation': item.get('rain', {}).get('3h', 0) # Handle missing rain safely
                })
        else:
            print(f"API failed for {city}. Generating realistic mock data for exam demonstration...")
            return generate_mock_data()
            
    return pd.DataFrame(weather_list)

def generate_mock_data():
    """Fallback generator if API key is invalid/missing."""
    np.random.seed(42)
    dates = [datetime.now() - timedelta(days=i) for i in range(30)]
    coords = {"Pune": (18.52, 73.85), "Mumbai": (19.07, 72.87), "Delhi": (28.61, 77.20), 
              "Bangalore": (12.97, 77.59), "Chennai": (13.08, 80.27)}
    
    data = []
    for city, (lat, lon) in coords.items():
        for d in dates:
            data.append({
                'City': city,
                'Lat': lat,
                'Lon': lon,
                'DateTime': d.strftime('%Y-%m-%d %H:%M:%S'),
                'Temp': np.random.normal(28 if city != 'Delhi' else 22, 5),
                'Humidity': np.random.uniform(40, 90),
                'WindSpeed': np.random.uniform(2, 15),
                'Precipitation': np.random.choice([0, 0, 0, 5, 12, 20])
            })
    return pd.DataFrame(data)

def main():
    # Fetch and load data
    df = fetch_weather_data()
    
    print("\nTask 4: Cleaning and Preprocessing...")
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df['Precipitation'] = df['Precipitation'].fillna(0)
    df['Date'] = df['DateTime'].dt.date # Extract just the date for aggregation
    print("Data cleaned. Shape:", df.shape)

    print("\nTask 5: Data Modeling (Averages, Max, Min)...")
    model_stats = df.agg({
        'Temp': ['mean', 'max', 'min'],
        'Humidity': ['mean'],
        'WindSpeed': ['mean'],
        'Precipitation': ['mean', 'max']
    }).round(2)
    print(model_stats)

    print("\nTask 7: Data Aggregation (Daily Statistics for Pune)...")
    pune_df = df[df['City'] == 'Pune']
    daily_stats = pune_df.groupby('Date')[['Temp', 'Precipitation']].mean().round(2)
    print(daily_stats.head())

    print("\nTask 6, 8, & 9: Generating Visualizations...")
    fig = plt.figure(figsize=(15, 10))

    # 1. Line Chart: Temperature trends (Task 6)
    ax1 = fig.add_subplot(221)
    for city in CITIES:
        city_data = df[df['City'] == city].groupby('Date')['Temp'].mean()
        ax1.plot(city_data.index, city_data.values, marker='o', label=city)
    ax1.set_title("Average Daily Temperature Trend")
    ax1.set_ylabel("Temp (°C)")
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend()

    # 2. Bar Plot: Precipitation levels (Task 6)
    ax2 = fig.add_subplot(222)
    city_precip = df.groupby('City')['Precipitation'].sum()
    ax2.bar(city_precip.index, city_precip.values, color='teal')
    ax2.set_title("Total Precipitation by City")
    ax2.set_ylabel("Precipitation (mm)")

    # 3. Geospatial Scatter Plot (Task 8)
    ax3 = fig.add_subplot(223)
    scatter = ax3.scatter(df['Lon'], df['Lat'], c=df['Temp'], cmap='coolwarm', s=200, alpha=0.7, edgecolors='k')
    for city in CITIES:
        city_coord = df[df['City'] == city].iloc[0]
        ax3.annotate(city, (city_coord['Lon'], city_coord['Lat']), xytext=(5,5), textcoords='offset points')
    ax3.set_title("Geospatial Weather Map (Color = Temp)")
    ax3.set_xlabel("Longitude")
    ax3.set_ylabel("Latitude")
    plt.colorbar(scatter, ax=ax3, label='Temperature (°C)')

    # 4. Correlation Heatmap (Task 9)
    ax4 = fig.add_subplot(224)
    corr_matrix = df[['Temp', 'Humidity', 'WindSpeed', 'Precipitation']].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='RdYlGn', vmin=-1, vmax=1, ax=ax4)
    ax4.set_title("Weather Attribute Correlation")

    plt.tight_layout()
    plt.savefig('weather_analysis_dashboard.png')
    print("-> Saved all plots as 'weather_analysis_dashboard.png'")
    print("\nExecution Complete!")

if __name__ == "__main__":
    main()

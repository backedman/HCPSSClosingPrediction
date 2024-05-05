import argparse
import openmeteo_requests
import requests_cache
from retry_requests import retry
from tqdm import tqdm
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, ComplementNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,recall_score,precision_score
from model import *

def add_snow_dates(df):
    # List of special dates
    snow_days = [
        "January 6, 2015",
        "January 26, 2015",
        "January 27, 2015",
        "February 26, 2015",
        "March 2, 2015",
        "March 4, 2015",
        "March 5, 2015",
        "March 6, 2015",
        "February 9, 2016",
        "February 16, 2016",
        "January 26, 2016",
        "January 27, 2016",
        "January 28, 2016",
        "January 29, 2016",
        "March 14, 2017",
        "December 9, 2017",
        "February 5, 2018",
        "February 7, 2018",
        "March 2, 2018",
        "March 22, 2018",
        "January 14, 2019",
        "February 11, 2019",
        "December 16, 2019"
    ]
    
    # Convert special dates to datetime format
    snow_days = pd.to_datetime(snow_days)

    # Convert dates in DataFrame to datetime format
    
    df['dt_date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    
    # Add a new column with 1 for special dates and 0 for others
    df['is_snow_date'] = df['dt_date'].isin(snow_days).astype(int)

    # Drop the "school_year" column
    df.drop(columns=['dt_date'], inplace=True)
    
    return df

def one_hot_encode(df):
    '''One hot encodes the weather_code columns for each location, but only for the columns relelated to snow'''
    
    # Define weather codes to be one-hot encoded
    weather_codes = list(range(1, 91))#[20,21,22, 24, 26, 28, 37, 38, 39, 70, 71, 72, 73, 74, 75, 83, 84, 85, 86, 87, 88, 89, 90]

    weather_code_columns = [col for col in df.columns if col.endswith('weather_code')]

    for column in weather_code_columns:
        # Extract the location name from the column name
        location = column.replace('_weather_code', '')
        
        # One-hot encode each weather code and convert to 1s and 0s
        for code in weather_codes:
            new_column_name = f"{location}_weather_code_{code}"
            df[new_column_name] = df[column].apply(lambda x: 1 if str(code) in str(x) else 0)
        
        df.drop(columns=[column], inplace=True)

    return df

def fetch_weather_data():
    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)
    print("here")

    # Define the date range
    start_date = "2015-01-01"
    end_date = "2019-12-31"

    # Define the locations with their coordinates
    locations = [
        ("annapolisjunction", (39.11907141644279, -76.77729698960306)),
        ("clarksville1", (39.1818477860658, -76.9262721420152)),
        #("clarksville2", (39.221489197242185, -76.97399400189961)),
        ("columbia", (39.20627079328966, -76.85824308567241)),
        #("columbia2", (39.22203612160417, -76.89279215083049)),
        #("columbia3", (39.18853056812141, -76.81436789249621)),
        ("cooksville", (39.31432186785972, -77.02078464571065)),
        #("cooksville2", (39.34506092587917, -77.01297405353533)),
        ("Dayton", (39.24173802113369, -76.98608152212104)),
        ("Elkridge", (39.197674494406456, -76.76590386783121)),
        ("EllicottCity1", (39.27807842792331, -76.82005766432094)),
        #("EllicottCity2", (39.255883942998366, -76.8665778946399)),
        #("EllicottCity3", (39.296679166629715, -76.80271986630537))
    
    ]

    # Define the weather variables you want to retrieve
    weather_variables = [
        "weather_code",
        "temperature_2m_max",
        "temperature_2m_min",
        "apparent_temperature_max",
        "apparent_temperature_min",
        "precipitation_sum",
        "rain_sum",
        "snowfall_sum",
        "precipitation_hours",
        "sunrise",
        #"sunset",
        "sunshine_duration",
        "daylight_duration",
        "wind_speed_10m_max",
        "wind_gusts_10m_max",
        #"wind_direction_10m_dominant"
        #"shortwave_radiation_sum",
        #"et0_fao_evapotranspiration"
    ]

    # Initialize a dictionary to store daily weather data
    weather_data = {"date": pd.date_range(start=start_date, end=end_date, freq='D')}

    # Iterate over locations and fetch weather data
    for location, coordinates in tqdm(locations):
        latitude, longitude = coordinates
        
        # Fetch weather data for the location
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "daily": ",".join(weather_variables)
        }
        responses = openmeteo.weather_api(url, params=params)
        
        # Process daily data
        for response in responses:
            daily = response.Daily()
            for ind, variable in enumerate(weather_variables):
                variable_values = daily.Variables(ind).ValuesAsNumpy()
                weather_data[f"{location}_{variable}"] = variable_values
        time.sleep(240/len(locations))  # Add a small delay to avoid overloading the server

    # Create a DataFrame from the weather data
    weather_df = pd.DataFrame(data=weather_data)

    # Save the DataFrame to a CSV file
    weather_df.to_csv("weather_data.csv", index=False)

    return weather_df





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch weather data.")
    parser.add_argument("--fetch_data", action="store_true", help="Fetch weather data.")
    parser.add_argument("--train", action="store_true", help="Fetch weather data.")
    args = parser.parse_args()
    snow_model = None
    
    if args.fetch_data:
        #weather_df = None
        weather_df = fetch_weather_data()
        
        #if weather_df == None:
        #weather_df = pd.read_csv("weather_data.csv")

        weather_df = add_snow_dates(weather_df)
        weather_df = one_hot_encode(weather_df)

        weather_df.to_csv("weather_data.csv", index=False)
        
    weather_df = pd.read_csv("weather_data.csv")
    
    if args.train:
        snow_model = SnowDayModel()
        snow_model.train_and_evaluate("weather_data.csv")
        snow_model.save_model('model.pkl')
        

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, accuracy_score, precision_score
import argparse
import openmeteo_requests
import requests_cache
from retry_requests import retry
import pickle
from datetime import datetime, date, timedelta



cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

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

class SnowDayModel:
    def __init__(self):
        self.model = None
        self.Test = None

    def train_model(self, weather_data_file):
        weather_df = pd.read_csv(weather_data_file)
        
        X = weather_df.drop(columns=['date', 'is_snow_date'])
        y = weather_df['is_snow_date']
        
        # Split the data into training and testing sets (50% train, 50% test)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21)
        
        self.Test = (X_test, y_test)

        # Train a Random Forest Classifier
        self.model = RandomForestClassifier()
        self.model.fit(X_train, y_train)

    def evaluate_model(self):
        X_test, y_test = self.Test
        
        # Predict probabilities of positive class (snow days)
        y_probs = self.model.predict_proba(X_test)[:, 1]

        # Adjust the threshold for classification to prioritize sensitivity
        threshold = 0.5  # Adjust threshold to capture 95% of positive instances
        print(threshold)

        # Classify instances based on adjusted threshold
        y_pred = (y_probs >= threshold).astype(int)

        # Evaluate the model for recall, accuracy, and precision
        sensitivity = recall_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        
        print(f"Sensitivity (Recall): {sensitivity:.2f}")
        print(f"Precision: {precision:.2f}")
        print(f"Accuracy: {accuracy}")
        
    def preprocess(self, df):
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
        
    def predict(self, input_date):
        
        # Initialize a dictionary to store daily weather data
        weather_data = {"date": pd.date_range(start=input_date, end=input_date, freq='D')}
        
        today = date.today()
        input_date_obj = datetime.strptime(input_date, "%Y-%m-%d").date()
        
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
        
        for location, coordinates in locations:
            latitude, longitude = coordinates
            
            print(date.today())
            print(input_date)
            
            url = None
            params = None

            #if the user asks for the current day, get data from the non-historical forecast for today
            if(input_date_obj == today):
                print('here')
                url = "https://api.open-meteo.com/v1/forecast"
                params = {
                    "latitude": latitude,
                    "longitude": longitude,
                    "daily": weather_variables,
                    "forecast_days": 1
                }
            #if the user asks for the previous day, get data from the non-historical forecast for yesterday
            elif(input_date_obj == today - timedelta(days=1)):
                url = "https://api.open-meteo.com/v1/forecast"
                params = {
                    "latitude": latitude,
                    "longitude": longitude,
                    "daily": weather_variables,
                    "forecast_days": 1,
                    "past_days": 1
                }
            #if the user asks for any other past day, get data from the archived forecast for the past days
            elif(input_date_obj < today):
                url = "https://archive-api.open-meteo.com/v1/archive"
                params = {
                    "latitude": latitude,
                    "longitude": longitude,
                    "start_date": input_date,
                    "end_date": input_date,
                    "daily": ",".join(weather_variables)
                }
            else: #user asks for future day, which we don't cover/predict for
                return [-1]    
            
                
                
            
            
            responses = openmeteo.weather_api(url, params=params)
            print(responses)
                    
            # Process daily data
            for response in responses:
                daily = response.Daily()
                for ind, variable in enumerate(weather_variables):
                    variable_values = daily.Variables(ind).ValuesAsNumpy()
                    weather_data[f"{location}_{variable}"] = variable_values
            
        # Create a DataFrame from the weather data
        weather_df = pd.DataFrame(data=weather_data)
        weather_df = self.preprocess(weather_df)
        
        X = weather_df.drop(columns=['date'])
        
        y_probs = self.model.predict_proba(X)[:, 1]
            
        return y_probs

    def train_and_evaluate(self, weather_data_file):
        self.train_model(weather_data_file)
        self.evaluate_model()

    def save_model(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump((self.model, self.Test), f)
            
    def load_model(self, file_path):
        with open(file_path, 'rb') as f:
            self.model, self.Test = pickle.load(f)

import argparse
import openmeteo_requests
import requests_cache
from retry_requests import retry
from tqdm import tqdm
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,recall_score

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
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    
    # Add a new column with 1 for special dates and 0 for others
    df['is_snow_date'] = df['date'].isin(snow_days).astype(int)

    # Calculate the cumulative count of snow days in the school year so far
    df['school_year'] = (weather_df['date'].dt.month >= 9) | (weather_df['date'].dt.month <= 5)
    df['cumulative_snow_days_school_year'] = weather_df.groupby(weather_df['school_year'].cumsum())['is_snow_date'].cumsum()

    # Drop the "school_year" column
    weather_df.drop(columns=['school_year'], inplace=True)
    
    return df

def one_hot_encode(df):

    # Identify weather code columns ending with 'weather_code'
    weather_code_columns = [col for col in df.columns if col.endswith('weather_code')]
    
    # One-hot encode each weather code column and convert to 1s and 0s
    for column in weather_code_columns:
        encoded_column = pd.get_dummies(df[column], prefix=column, drop_first=True)
        df.drop(columns=[column], inplace=True)
        df = pd.concat([df, encoded_column], axis=1)
        df = df.astype(int)
    
    return df

def fetch_weather_data():
    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    # Define the date range
    start_date = "2015-01-01"
    end_date = "2019-12-31"

    # Define the locations with their coordinates
    locations = [
        ("annapolisjunction", (39.11907141644279, -76.77729698960306)),
        ("clarksville1", (39.1818477860658, -76.9262721420152)),
        ("clarksville2", (39.221489197242185, -76.97399400189961)),
        ("columbia", (39.20627079328966, -76.85824308567241)),
        ("columbia2", (39.22203612160417, -76.89279215083049)),
        ("columbia3", (39.18853056812141, -76.81436789249621)),
        ("cooksville", (39.31432186785972, -77.02078464571065)),
        ("cooksville2", (39.34506092587917, -77.01297405353533)),
        ("Dayton", (39.24173802113369, -76.98608152212104)),
        ("Elkridge", (39.197674494406456, -76.76590386783121)),
        ("EllicottCity1", (39.27807842792331, -76.82005766432094)),
        ("EllicottCity2", (39.255883942998366, -76.8665778946399)),
        ("EllicottCity3", (39.296679166629715, -76.80271986630537))
    
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
        "wind_direction_10m_dominant"
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


def train_naive_bayes_model(weather_data_file):
    # Load the weather data from CSV file
    weather_df = pd.read_csv(weather_data_file)

    # Split the dataset into features (X) and target (y)
    X = weather_df.drop(columns=['date', 'is_snow_date'])
    y = weather_df['is_snow_date']

    # Split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Naive Bayes model
    naive_bayes_model = GaussianNB()
    naive_bayes_model.fit(X_train, y_train)

    # Predict probabilities of positive class (snow days)
    y_probs = naive_bayes_model.predict_proba(X_test)[:, 1]

    # Adjust the threshold for classification to prioritize sensitivity
    threshold = np.percentile(y_probs, 95)  # Adjust threshold to capture 95% of positive instances

    # Classify instances based on adjusted threshold
    y_pred = (y_probs >= threshold).astype(int)

    # Evaluate the model
    sensitivity = recall_score(y_test, y_pred)
    accuracy = accuracy_score(y_test,y_pred)
    print(f"Sensitivity (Recall): {sensitivity:.2f}")
    print(f"accuracy: {accuracy}")

    return naive_bayes_model





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch weather data.")
    parser.add_argument("--fetch_data", action="store_true", help="Fetch weather data.")
    args = parser.parse_args()

    if args.fetch_data:
        weather_df = None
        #weather_df = fetch_weather_data()
        
        if weather_df == None:
            weather_df = pd.read_csv("weather_data.csv")

        weather_df = add_snow_dates(weather_df)
        weather_df = one_hot_encode(weather_df)

        weather_df.to_csv("weather_data.csv", index=False)

        # Example usage
        model = train_naive_bayes_model("weather_data.csv")

        # Preprocess the weather data for prediction
        X_pred = weather_df.drop(columns=['date', 'is_snow_date'])

        # Predict probabilities of snow days using the trained model
        y_probs = model.predict_proba(X_pred)[:, 1]

        # Predict snow days using a threshold (e.g., 0.5)
        threshold = 0.5
        y_pred = (y_probs >= threshold).astype(int)

        # Create a DataFrame with the date, predicted snow day, actual snow day, and confidence
        predictions_df = pd.DataFrame({
            'date': weather_df['date'],
            'predicted_snow_day': y_pred,
            'actual_snow_day': weather_df['is_snow_date'],
            'confidence': y_probs
        })


        # Save the predictions DataFrame to a new CSV file
        predictions_df.to_csv("predictions.csv", index=False)
        

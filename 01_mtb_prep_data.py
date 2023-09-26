import pickle
import os
from datetime import datetime, timedelta
import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, ParameterGrid, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from scipy.stats import ttest_ind
from xgboost import XGBClassifier
import eli5
import shap
import matplotlib as mpl
import matplotlib.dates as mdates
from sklearn.utils import resample
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import boto3
import json
import time
import os
from cryptography.fernet import Fernet

warnings.filterwarnings('ignore')

# from googleapiclient.discovery import build
# from google_auth_oauthlib.flow import InstalledAppFlow

# SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']
# CREDENTIALS_PATH = '/Users/jacksonburton/Documents/tech_projects/full_projects/creds/g_sheets_mtb_weather_creds.json'
# SPREADSHEET_ID = '1MgF0cVUuf6L2yemHfydQGERZt8fe7IirK_eUr-OisPk'
# RANGE_NAME = 'Sheet1!A:D'
# flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
# creds = flow.run_local_server(port=0)
# service = build('sheets', 'v4', credentials=creds)
# sheet = service.spreadsheets()
# result = sheet.values().get(spreadsheetId=SPREADSHEET_ID, range=RANGE_NAME).execute()
# values = result.get('values', [])

# keys = values[0]
# data = {key: [] for key in keys}

# for row in values[1:]:
#     for i, value in enumerate(row):
#         if keys[i] == 'AVG WIND' or keys[i] == 'TOTAL PRECIPITATION':
#             data[keys[i]].append(float(value))
#         else:
#             data[keys[i]].append(value)



# def lambda_handler(event, context):
#     # Define the S3 bucket and file
#     bucket = 'mtb-trail-condition-predictions'
#     key = 'daily_weather.json'

#     # Create an S3 client
#     s3 = boto3.client('s3')

#     # Read the file from S3
#     response = s3.get_object(Bucket=bucket, Key=key)
#     file_content = response['Body'].read().decode('utf-8')
#     data = json.loads(file_content)

#     response = s3.get_object(Bucket=bucket, Key=key)
#     file_content = response['Body'].read().decode('utf-8')
#     data = json.loads(file_content)

#     return data



# # Create the DataFrame
# yesterday_weather= pd.DataFrame(data_yesterday)
# yesterday_weather['MAX TEMPERATURE'] = pd.to_numeric(yesterday_weather['MAX TEMPERATURE'])
# yesterday_weather

# %% [markdown]
# # Read in Cora Data, Process

# %%
noaa_id = 'NOAA_3392043'
lookback_days_list = [2, 3, 5]


# %%
# df = pd.read_csv('data/cora-history-raw.csv')

# Create a client connection to S3
s3_client = boto3.client('s3')

# Specify the bucket name and key for the CSV file
bucket_name = 'mtb-trail-condition-predictions'
file_key = 'data/cora-history-raw.csv'

# Use the S3 client to retrieve the file object
obj = s3_client.get_object(Bucket=bucket_name, Key=file_key)

# Read the file's contents into a Pandas DataFrame
print("Reading in file from S3 bucket")
df = pd.read_csv(obj['Body'])
# print(df.head(5))
df['date_clean'] = df['date'].str[:10]
df["date_clean"] = pd.to_datetime(df["date_clean"])
data = df.copy().sort_values(by="date", ascending=False)
data.head(1)

# %%
# Take the latest update in the day to classify the whole day
# Each day trails will often have multiple updates, open the day as open but then it rains, so it closes
# something to keep in mind when building the methodology
data_grouped = data.groupby("date_clean", as_index=False).first()
data_grouped = data_grouped.sort_values(by="date_clean", ascending=True)

# %%
# Create a new dataframe with all possible dates
all_dates = pd.date_range(start=data['date_clean'].min(), end=data['date_clean'].max(), freq='D')
all_dates_df = pd.DataFrame({'date_clean': all_dates})
all_dates_df.head(1)

# %%
# Create a new dataframe with all possible dates and trails
all_dates = pd.date_range(start=data['date_clean'].min(), end=data['date_clean'].max(), freq='D')
all_trails = data['trail'].unique()
all_dates_df = pd.DataFrame({
    'date_clean': pd.date_range(start=data['date_clean'].min(), end=data['date_clean'].max(), freq='D').repeat(len(all_trails)),
    'trail': list(all_trails) * len(all_dates)
})

all_dates_df['date_clean'] = pd.to_datetime(all_dates_df['date_clean'])
data = data.sort_values('date_clean')

# Merge the two dataframes, filling in missing values with the latest status available
merged_df = pd.merge_asof(all_dates_df, data, on='date_clean', by='trail', direction='backward')

# Forward fill the 'status' column to fill in any missing values
merged_df['status'] = merged_df['status'].fillna(method='ffill')

# Remove any unnecessary columns and sort by date_clean
trail_df = merged_df[['date_clean', 'trail', 'status']].sort_values('date_clean')

trail_df.sort_values(['trail','date_clean']).head(5)

# %% [markdown]
# # Load in NOAA Weather Data, Process

# %%
# weather_df = pd.read_csv('data/' +noaa_id + '.csv')

file_key = 'data/NOAA_3392043.csv'
obj = s3_client.get_object(Bucket=bucket_name, Key=file_key)
weather_df = pd.read_csv(obj['Body'])


# print(weather_df['DATE'].min())
# print(weather_df['DATE'].max())

# %% [markdown]
# ## Some Stations have more missing data than others
# - Not sure why this is, data is just corrupted and/or missing certain days at some stations

# %%
# Subset the DataFrame to include only the relevant columns
cols = ['NAME', 'AWND', 'PRCP', 'SNOW', 'TMAX', 'DATE']
df_subset = weather_df[cols]

# Get the max date from the DataFrame
max_date = df_subset['DATE'].max()

# Filter the DataFrame to only include rows with the max date
df_max_date = df_subset[df_subset['DATE'] == max_date]

# Group the DataFrame by the categorical column and count the missing values
grouped = df_max_date.groupby('NAME')
missing_counts = grouped.apply(lambda x: x.isnull().sum())

# Find the group with the least missing values
min_missing_index = missing_counts.sum(axis=1).idxmin()

# Retrieve the DataFrame for the group with the least missing values
df_min_missing = grouped.get_group(min_missing_index)
df_min_missing['NAME'].unique()


weather_df_clean = weather_df[weather_df['NAME'] == min_missing_index]

weather_df_clean = weather_df_clean[['DATE', 'STATION', 'NAME', 'AWND', 'PRCP','SNOW','TAVG',
                                     'TMAX','TMIN']]

weather_df_clean.head(1)


# %%
# Subset the DataFrame to include only the relevant columns
cols = ['NAME', 'AWND']
df_subset = weather_df[cols]

# Group the DataFrame by the categorical column and count the missing values
grouped = df_subset.groupby('NAME')
missing_counts = grouped.apply(lambda x: x.isnull().sum())

# Find the group with the least missing values
min_missing_index = missing_counts.sum(axis=1).idxmin()

# Retrieve the DataFrame for the group with the least missing values
df_min_missing = grouped.get_group(min_missing_index)
df_min_missing['NAME'].unique()

# weather_df['NAME'].unique()

weather_df_clean = weather_df[weather_df['NAME'] == 'CINCINNATI NORTHERN KENTUCKY INTERNATIONAL AIRPORT, KY US']

weather_df_clean = weather_df_clean[['DATE', 'STATION', 'NAME', 'AWND', 'PRCP','SNOW','TAVG',
                                     'TMAX','TMIN']]

weather_df_clean.head(1)

# %% [markdown]
# ## Check for Missing Values

# %%
# Count the number of missing values in each column
missing_value_counts = weather_df_clean.isnull().sum()

# Print the number of missing values in each column
# print(missing_value_counts[missing_value_counts > 0])

# %%
weather_df_clean = weather_df_clean[weather_df_clean['DATE'] < '2023-03-17']
# print(len(weather_df_clean))

# %%
# weather_df_clean.columns

# %%
weather_trim = weather_df_clean[['DATE', 'AWND', 'PRCP',
       'SNOW', 'TAVG', 'TMAX', 'TMIN']]
weather_trim = weather_trim.rename(columns={'DATE': 'date_clean'})
weather_trim["date_clean"] = pd.to_datetime(weather_trim["date_clean"])
weather_trim.head(1)

# %% [markdown]
# # Merge NOAA Weather and Trail Data

# %% [markdown]
# ## Define Open/Caution/Freeze as Target Variable

# %%
model_df = trail_df.merge(weather_trim, on = 'date_clean', how = 'inner')
model_df['PRCP'] = model_df['PRCP'] + model_df['SNOW']
model_df = model_df.drop(columns=['TMIN', 'TAVG', 'SNOW'])

# Create new columns 
model_df = model_df.sort_values(['trail', 'date_clean']) # this sort is crucial to the logic below
for i in lookback_days_list:
    for col in ['PRCP', 'TMAX']:
        if col == 'PRCP':
            # Calculate the cumulative sum of 'PRCP' for the past X days including today
            model_df[f'{col}_{i}d'] = model_df[col].rolling(window=i, min_periods=1).sum()
        else:
            # Calculate the average of 'AWND' and 'TAVG' for the past X days including today
            model_df[f'{col}_{i}d'] = model_df[col].rolling(window=i, min_periods=1).mean()

# Replace missing values with 0
model_df.fillna(0, inplace=True)

# Create target column with 3 categories: open, caution, closed/freeze
conditions = [
    model_df['status'].isin(['open', 'caution','freeze']),
    model_df['status'].isin(['closed']),
]
choices = [1, 0]
model_df['target'] = np.select(conditions, choices, default='closed/freeze').astype(int)

# Drop unnecessary columns
model_df.drop(columns=['status'], inplace=True)

# Encode target variable as categorical
model_df['target'] = pd.Categorical(model_df['target'])

def load_key():
    return open("/Users/jacksonburton/Documents/tech_projects/key.key", "rb").read()

# Load the key
key = load_key()
f = Fernet(key)

with open("/Users/jacksonburton/Documents/tech_projects/encrypted_api_key.key", "rb") as api_key_file:  
    encrypted_api_key = api_key_file.read()

decrypted_api_key = f.decrypt(encrypted_api_key).decode()
api_key = decrypted_api_key

# %% [markdown]
# # METHODOLOGY: % Probability X Rain Inches
# - Since future weather data is uncertain, lets try taking the probability of rain TIMES the inches expected
# READ in Long/Lat of trails
df_trail_locations = pd.read_csv("data/trail_locations.csv")
exclude = "minutely,hourly,alerts"
days=7
pickle_file = 'data/weather_data.pickle'

# Check if pickle file exists and is from today
if os.path.exists(pickle_file) and datetime.fromtimestamp(os.path.getmtime(pickle_file)).date() == datetime.now().date():
    with open(pickle_file, 'rb') as f:
        future_weather_all_trails = pickle.load(f)
    print("Already have today's future data. Loading from pickle file.")
else:
    # Initialize an empty DataFrame to store all future weather data
    future_weather_all_trails = pd.DataFrame()

    # Loop through each trail in df_trail_locations
    for index, row in df_trail_locations.iterrows():
        lat = row['Latitude']
        lon = row['Longitude']
        trail = row['Trail']
        
        url = f"https://api.openweathermap.org/data/3.0/onecall?lat={lat}&lon={lon}&exclude={exclude}&appid={api_key}&units=imperial"
        response = requests.get(url).json()
        forecast = response['daily']
        data = []
        
        for day in forecast:
            date = datetime.fromtimestamp(day['dt']).strftime('%Y-%m-%d')
            max_temp_f = day['temp']['max']
            avg_wind = day['wind_speed']
            total_precip_mm = day['rain'] if 'rain' in day else 0
            total_precip_in = total_precip_mm / 25.4
            precip_prob = day['pop']
            data.append([trail, date, max_temp_f, avg_wind, total_precip_in, precip_prob])

        weather_data = pd.DataFrame(data, columns=['Trail', 'DATE', 'MAX TEMPERATURE', 'AVG WIND', 'TOTAL PRECIPITATION', 'PRECIPITATION PROBABILITY'])
        future_weather_all_trails = pd.concat([future_weather_all_trails, weather_data])

    # Save to pickle
    with open(pickle_file, 'wb') as f:
        pickle.dump(future_weather_all_trails, f)
    print("Did not have today's future data, loading API data and saving to pickle file.")

# Your existing code for modifying weather data
future_weather_all_trails['TOTAL PRECIPITATION RAW'] = future_weather_all_trails['TOTAL PRECIPITATION'].copy()
future_weather_all_trails['ADJUSTED PRECIPITATION PROBABILITY'] = (future_weather_all_trails['PRECIPITATION PROBABILITY'] * 1.00).clip(upper=1)
# future_weather_all_trails['ADJUSTED PRECIPITATION PROBABILITY'] = future_weather_all_trails['PRECIPITATION PROBABILITY'].apply(lambda x: x ** 2) ## Power Transform, keeps higher values high but minimizes lower values
future_weather_all_trails['TOTAL PRECIPITATION'] = future_weather_all_trails['TOTAL PRECIPITATION'] * future_weather_all_trails['ADJUSTED PRECIPITATION PROBABILITY']
future_weather_all_trails.loc[future_weather_all_trails['TOTAL PRECIPITATION'] < 0.01, 'TOTAL PRECIPITATION'] = 0

print("FUTURE WEATHER ALL TRAILS ---------------")
print(future_weather_all_trails.head(30))

# Final DataFrame
future_weather = future_weather_all_trails[['Trail', 'DATE', 'MAX TEMPERATURE', 'AVG WIND', 'TOTAL PRECIPITATION']]

# %%
def get_weather_data(lat, lon, date, api_key):
    base_url = "https://api.openweathermap.org/data/3.0/onecall/day_summary"
    params = {
        'lat': lat,
        'lon': lon,
        'date': date,
        'appid': api_key,
        'units': 'imperial'  # For Fahrenheit
    }
    response = requests.get(base_url, params=params)
    return response.json()


pickle_file = 'data/historical_one_week_all_trails.pickle'

if os.path.exists(pickle_file) and datetime.fromtimestamp(os.path.getmtime(pickle_file)).date() == datetime.now().date():
    with open(pickle_file, 'rb') as f:
        historical_one_week_all_trails = pickle.load(f)
    print("Already have today's historical data. Loading from pickle file.")
else:
    # Initialize an empty DataFrame
    historical_one_week_all_trails = pd.DataFrame()

    # Loop through each trail in df_trail_locations
    for index, row in df_trail_locations.iterrows():
        lat = row['Latitude']
        lon = row['Longitude']
        trail = row['Trail']
        data = []
        
        # Loop through last 7 days
        for i in range(1, 8):
            date = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
            response = get_weather_data(lat, lon, date, api_key)

            try:
                temp_max = response['temperature']['max']
                wind_speed = response['wind']['max']['speed']
                precipitation = response['precipitation']['total'] * 0.0393701
                
                data.append([trail, date, temp_max, wind_speed, precipitation])
            except KeyError:
                print(f"Error for {date}: {response.get('message', 'Unknown error')}")

        historical_data = pd.DataFrame(data, columns=['Trail', 'DATE', 'MAX TEMPERATURE', 'AVG WIND', 'TOTAL PRECIPITATION'])
        historical_one_week_all_trails = pd.concat([historical_one_week_all_trails, historical_data])

    # Save to pickle
    with open(pickle_file, 'wb') as f:
        pickle.dump(historical_one_week_all_trails, f)
    print("Did not have today's historical data, pulling API data and saving to pickle file.")


# weather_df_deploy = weather_trim[['date_clean', 'TMAX', 'AWND', 'PRCP']]

# column_names = {'date_clean': 'DATE', 'TMAX': 'MAX TEMPERATURE', 'AWND': 'AVG WIND','PRCP' : 'TOTAL PRECIPITATION'}
# weather_df_deploy = weather_df_deploy.rename(columns=column_names)
# weather_df_deploy.head(1)

print("hist one week all trails")
print(historical_one_week_all_trails.head(5))
print(historical_one_week_all_trails.columns)

print("future weather")
print(future_weather.head(5))
print(future_weather.columns)


weather_append = pd.concat([historical_one_week_all_trails,
    # historical_one_week, 
# yesterday_weather, weather_df_deploy
                            future_weather])
weather_append = weather_append.drop_duplicates(subset=['DATE', 'Trail'], keep='first')
weather_append['DATE'] = pd.to_datetime(weather_append['DATE'])
weather_append['DATE'] = weather_append['DATE'].dt.date
print("PRINTING NEW WEATHER DATA #######")
print(weather_append.head(25))



# # Visualize Recent Weather Data and Future Forecast
weather_sorted = weather_append.sort_values(by='DATE', ascending=False)
weather_sorted.set_index('DATE', inplace=True)
print(weather_sorted.head(15))
# %%
# # Plot TOTAL PRECIPITATION
# plt.figure(figsize=(15, 3))
# plt.bar(weather_sorted.index, weather_sorted['TOTAL PRECIPITATION'], color='skyblue')
# plt.xticks(rotation=45)
# plt.title('TOTAL PRECIPITATION over Time')
# plt.xlabel('DATE')
# plt.ylabel('TOTAL PRECIPITATION')
# plt.grid(True)
# plt.show(block=False)


# create a dictionary that maps the old column names to the new column names
column_names = {'Trail': 'trail', 'DATE': 'date_clean', 'MAX TEMPERATURE': 'TMAX', 'AVG WIND': 'AWND', 'TOTAL PRECIPITATION': 'PRCP'}

# rename the columns using the rename() method with the column_names dictionary
weather_sorted = weather_sorted.reset_index().rename(columns=column_names)
weather_sorted.head(1)

# %%
len(weather_sorted)

# %%
model_df['date_clean'] = pd.to_datetime(model_df['date_clean'])
model_df['date_clean'] = model_df['date_clean'].dt.date

print("checkpoint bogie")
print(weather_sorted.head(30))

# weather_data_main = pd.concat([model_df, new_df], axis=0)
weather_data_main = weather_sorted.copy()
print(weather_data_main.columns)
weather_data_main = weather_data_main.drop_duplicates(subset=['date_clean', 'trail'])

print("checkpoint zdog")
print(weather_data_main.head(30))

print("model_df date min")
print(model_df['date_clean'].min())
print("model_df date max")
print(model_df['date_clean'].max())
# print("predict_df date min")
# print(new_df['date_clean'].min())
# print("predict_df date max")
# print(new_df['date_clean'].max())

# ## QA NOTE: Stopping At March 16th, 2023 because that's when CORA history ends

# weather_data_main.sort_values('date_clean', ascending = False).head(15)['date_clean'].unique()

# %% [markdown]
# ### Check to see if we have necessary dates 

# # %%
# if weather_data_main['date_clean'].max() < new_df['date_clean'].min():
#     print("WARNING missing weather data")
# else:
#     print("GOOD we have the proper data needed")


# # Feature Engineering # Define Lookbacks (# of days for each feature)
weather_data_main = weather_data_main.sort_values(['trail','date_clean'])

for i in lookback_days_list:
    for col in ['PRCP', 'TMAX']:
        if col == 'PRCP':
            # Calculate the cumulative sum of 'PRCP' for the past X days including today
            weather_data_main[f'{col}_{i}d'] = weather_data_main[col].rolling(window=i, min_periods=1).sum()
        else:
            # Calculate the average of 'AWND' and 'TAVG' for the past X days including today
            weather_data_main[f'{col}_{i}d'] = weather_data_main[col].rolling(window=i, min_periods=1).mean()

# Replace missing values with 0
weather_data_main.fillna(0, inplace=True)

# %%
# Filter to future dates

today = pd.Timestamp.today().normalize() # Get today's date
future_dates_mask = weather_data_main['date_clean'] >= today
weather_data_main_future = weather_data_main[future_dates_mask]


# %%
datetime_cols = model_df.select_dtypes(include=[np.datetime64, 'datetime', 'datetime64']).columns.tolist()
model_df['date_clean'] = model_df['date_clean'].astype('datetime64[ns]')
model_df['target'] = model_df['target'].astype('int64')

## Manually override "bad" trail status conditions
# Our training data is not perfect, often times a trail steward could be a day or so late to update the facebook page
# let's override trails to be CLOSED when there is at least 0.5 inches of rain

prcp_override = 0.50
prcp_override_2_days = 1.25
prcp_override_3_days = 2.25


# Update 'target' based on condition
model_df['target'] = np.where(model_df['PRCP'] >= prcp_override, 0, model_df['target'])
model_df['target'] = np.where(model_df['PRCP_2d'] >= prcp_override_2_days, 0, model_df['target'])
model_df['target'] = np.where(model_df['PRCP_3d'] >= prcp_override_3_days, 0, model_df['target'])


### Try PCA on PRCP + AWND, PRCP + TEMPERATURE

cols_to_keep =  ['date_clean', 'trail', 'target', 'PRCP', 'PRCP_2d', 'PRCP_3d', 'PRCP_5d', 'PC1_1d', 'PC1_5d'] 

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def add_principal_components(df):
    # Function to perform PCA on given columns and return principal components
    def perform_pca(columns, n_components=1):
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df[columns])
        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(scaled_data)
        
        # Print explained variance ratios
        print(f"Explained variance ratios for {columns}: {pca.explained_variance_ratio_}")
        
        return principal_components

    # Perform PCA on the first set of columns
    principal_components_1 = perform_pca(['AWND', 'TMAX', 'PRCP'])
    principal_df_1 = pd.DataFrame(data=principal_components_1, columns=['PC1_1d'])
    
    # Perform PCA on the second set of columns
    principal_components_2 = perform_pca(['TMAX_5d', 'PRCP_5d'])
    principal_df_2 = pd.DataFrame(data=principal_components_2, columns=['PC1_5d'])
    
    # Concatenate original DataFrame and principal components
    final_df = pd.concat([df.reset_index(drop=True), principal_df_1.reset_index(drop=True), principal_df_2.reset_index(drop=True)], axis=1)
    keep_columns = [col for col in ['date_clean', 'trail', 'target', 'PRCP', 'PRCP_2d', 'PRCP_3d', 'PRCP_5d', 'PC1_1d', 'PC1_5d'] if col in df.columns]
    final_df = final_df[keep_columns]

    return final_df

# Apply function
model_df_with_pca = add_principal_components(model_df)
weather_data_main_future_with_pca = add_principal_components(weather_data_main_future)


def keep_selected_columns(df, cols_to_keep):
    # Filter columns that exist in the DataFrame
    filtered_columns = [col for col in cols_to_keep if col in df.columns]
    # Return DataFrame with selected columns
    return df[filtered_columns]


####### Write Out Data

cols_to_keep = ['date_clean', 'trail', 'target', 'PRCP', 
# 'PRCP_2d', 
'PRCP_3d', 
'PRCP_5d', 
# 'TMAX', 
# 'AWND',
'TMAX_5d'
]


model_df = keep_selected_columns(model_df, cols_to_keep)
weather_data_main_future = keep_selected_columns(weather_data_main_future, cols_to_keep)

model_df.to_csv('data/01_mtb_model_df_out.csv')

weather_data_main_future.to_csv('data/01_mtb_weather_data_main_future_out.csv')

print(weather_data_main_future.sort_values(['trail','date_clean']).head(15))


# model_df_with_pca.to_csv('data/01_mtb_model_df_out.csv')
# weather_data_main_future_with_pca.to_csv('data/01_mtb_weather_data_main_future_out.csv')

print("01 Script Complete")
############################
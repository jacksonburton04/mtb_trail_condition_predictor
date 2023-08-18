import pickle
import os
from datetime import datetime, timedelta
import requests
import pandas as pd
import numpy as np
import seaborn as sns
import warnings
import datetime as dt
from datetime import date
from matplotlib.colors import ListedColormap, BoundaryNorm
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# %%
weather_data_main_future = pd.read_csv('data/01_mtb_weather_data_main_future_out.csv')

# %%
weather_data_main_future.columns

# %%
import joblib

# Load the dictionary of models
trail_models = joblib.load('data/02_trail_models.joblib')

# %% [markdown]
# # Make Predictions

# %%
# Function to classify confidence level
def classify_confidence_level(confidence):
    if confidence >= 0.90:
        return 'OPEN - VERY LIKELY'
    elif confidence >= 0.75:
        return 'OPEN - LIKELY'
    elif confidence >= 0.50:
        return 'UNSURE - LEANING OPEN'
    elif confidence >= 0.25:
        return 'UNSURE - LEANING CLOSED'
    elif confidence >= 0.10:
        return 'CLOSED - LIKELY'
    elif confidence < 0.10:
        return 'CLOSED - VERY LIKELY'
    else:
        return 'ERROR'

trained_features_order = ['AWND', 'PRCP', 'TMAX', 
'PRCP_2d', 'TMAX_2d', 
'PRCP_3d', 'TMAX_3d', 
# 'PRCP_4d', 'TMAX_4d', 
 'PRCP_5d', 'TMAX_5d' 
#  'PRCP_6d', 'TMAX_6d', 
#  'PRCP_7d', 'TMAX_7d'
 ]

for trail, model in trail_models.items():
    trail_future_indices = weather_data_main_future["trail"] == trail
    trail_X_future = weather_data_main_future[trail_future_indices]
    model_features = [column for column in trained_features_order if column in trail_X_future.columns]

    missing_features = set(trained_features_order) - set(model_features)
    if missing_features:
        print(f"Trail: {trail}, Missing features: {missing_features}")

# %%
predictions = []

# Make sure 'date_clean' column is present in weather_data_main_future
weather_data_main_future['date_clean'] = pd.to_datetime(weather_data_main_future['date_clean'])



for trail, model in trail_models.items():
    trail_future_indices = weather_data_main_future["trail"] == trail
    trail_X_future = weather_data_main_future[trail_future_indices]

    # Extract model features dynamically in correct order (all columns except "trail" and "date_clean")
    model_features = [column for column in trained_features_order if column in trail_X_future.columns]

    if not trail_X_future.empty:
        # Predict probability of the positive class instead of predicting class
        y_pred_proba = model.predict_proba(trail_X_future[model_features])[:, 1]
        confidence_level_class = np.vectorize(classify_confidence_level)(y_pred_proba)

        trail_predictions = pd.DataFrame({"Date": trail_X_future['date_clean'],
                                          "CORA Trail": trail, 
                                          "Chance of Being Open": y_pred_proba, 
                                          "Classification": confidence_level_class})
        predictions.append(trail_predictions)

predictions_df = pd.concat(predictions, ignore_index=True)
predictions_df


# %% [markdown]
# # Save & Write Out Predictions

# %%


# Create separate variables for the long and short date formats
now = dt.datetime.now()
execution_date_long = now.strftime('%Y-%m-%d %H:%M:%S')
execution_date = now.strftime('%Y-%m-%d')

results_df = predictions_df.copy()
results_df['Date'] = pd.to_datetime(results_df['Date']).dt.date
results_df['execution_date_long'] = execution_date_long
results_df['execution_date'] = execution_date

all_results = pd.DataFrame(columns=['Date', 'CORA Trail', 'Chance of Being Open', 'Classification', 'execution_date_long', 'execution_date'])

# Check if the output file exists
try:
    output_df = pd.read_csv('data/output_file_cora.csv')
except FileNotFoundError:
    # If the file doesn't exist, create a new file with headers
    all_results.to_csv('data/output_file_cora.csv', index=False)
    output_df = pd.read_csv('data/output_file_cora.csv')

all_results = all_results.append(results_df, ignore_index=True)

# Append the current run's results to the output file
output_df = output_df.append(results_df, ignore_index=True)

# Save the updated output file
output_df.to_csv('data/output_file_cora.csv', index=False)

output_df.head(3)

# %% [markdown]
# # Visualize Predicted Open/Closed Trails into future

# %%
output_df_no_dups = output_df.sort_values('execution_date_long', ascending = False).drop_duplicates(subset=['Date', 'CORA Trail'], keep='first')
# output_df_no_dups[output_df_no_dups['CORA Trail'] == 'Devou Park'].sort_values('execution_date_long', ascending = False)

# %%


# %% [markdown]
# # Primary Data Viz

# %%
# No dups, take most recent prediction
output_df_no_dups = output_df.sort_values('execution_date_long', ascending = False).drop_duplicates(subset=['Date', 'CORA Trail'], keep='first')

# Convert 'Date' to datetime and sort
output_df_no_dups['Date'] = pd.to_datetime(output_df_no_dups['Date'], errors='coerce')

# Filter 'Date' by today or greater
today = pd.to_datetime(date.today())


# Subtract one day to get yesterday's date
yesterday = today - timedelta(days=1)

output_df_no_dups = output_df_no_dups[output_df_no_dups['Date'] >= today]

# Calculate mean 'Chance of Being Open' for each trail on each date
output_df_no_dups = output_df_no_dups.groupby(['Date', 'CORA Trail'], as_index=False).mean()

# Multiply 'Chance of Being Open' by 100
output_df_no_dups['Chance of Being Open'] = output_df_no_dups['Chance of Being Open'] * 100

# Create a new column 'Date_Str' for string formatted dates
output_df_no_dups['Date_Str'] = output_df_no_dups['Date'].dt.strftime('%A, %m-%d')

# # Calculate average 'Chance of Being Open' for each trail
# trail_avgs = output_df_no_dups.groupby('CORA Trail')['Chance of Being Open'].mean().sort_values(ascending=False)

# # Pivot the DataFrame using 'Date' instead of 'Date_Str'
# pivot_df = output_df_no_dups.pivot(index='Date', columns='CORA Trail', values='Chance of Being Open')

# # Reorder the columns based on their average 'Chance of Being Open'
# pivot_df = pivot_df[trail_avgs.index]
# print(pivot_df)

# ## WRITE OUT FOR VISUALIZE SCIPT
# pivot_df.to_csv('03_preds_viz.csv')

# Pivot the DataFrame using 'Date' instead of 'Date_Str'
pivot_df = output_df_no_dups.pivot(index='Date', columns='CORA Trail', values='Chance of Being Open')

# Sort the columns alphabetically by 'CORA Trail'
pivot_df = pivot_df.reindex(sorted(pivot_df.columns), axis=1)

print(pivot_df)

## WRITE OUT FOR VISUALIZE SCRIPT
pivot_df.to_csv('data/03_preds_viz.csv')

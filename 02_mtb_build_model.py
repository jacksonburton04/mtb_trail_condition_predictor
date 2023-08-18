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
from sklearn.model_selection import StratifiedKFold
import joblib


warnings.filterwarnings('ignore')

model_df = pd.read_csv('data/01_mtb_model_df_out.csv')
model_df = model_df.drop(model_df.columns[0], axis=1)

# %% [markdown]
# # Build Model
# - 1 is Open, 0 is Closed

print("data loaded")
model_df_input = model_df.copy()
unique_trails = model_df_input["trail"].unique()

# Drop Columns Not Needed for Modeling
if 'date_clean' in model_df_input.columns:
    model_df_input = model_df_input.drop(columns=['date_clean'])

# Define features and target
features = model_df_input.drop('target', axis=1)
target = model_df_input['target']

# Perform a train-test split
features_train, features_test, target_train, target_test = train_test_split(
    features, target, test_size=0.25, random_state=42)

# Further split the training set into a training and validation set
features_train, features_val, target_train, target_val = train_test_split(
    features_train, target_train, test_size=0.25, random_state=42)

# Dictionary to hold models for each trail
trail_models = {}

# Initialize lists to store data for final dataframes
all_feature_importances = []
shap_values_all = []
model_evaluations = []

print("modeling started, will take a while")
# New Param Grid 7/21
param_dist = {
    'n_estimators': range(50, 200, 50),
    'max_depth': range(2, 6),
    'min_child_weight': range(7, 35),
    'gamma': [i/10.0 for i in range(0, 10)],
    'subsample': [i/10.0 for i in range(3, 9)],
    'colsample_bytree': [i/10.0 for i in range(7, 11)],
    'learning_rate': [0.01, 0.05, 0.1],
    'reg_lambda': [0, 1, 2, 3],
    'reg_alpha': [0, 1, 2, 3]
}


# Train and select the best model for each trail
for trail in unique_trails:
    # Prepare train and validation data for the specific trail
    trail_mask_train = features_train["trail"] == trail
    trail_X_train = features_train.loc[trail_mask_train, :].drop(columns=['trail'])
    trail_y_train = target_train.loc[trail_mask_train]

    trail_mask_val = features_val["trail"] == trail
    trail_X_val = features_val.loc[trail_mask_val, :].drop(columns=['trail'])
    trail_y_val = target_val.loc[trail_mask_val]

    # Create a XGBClassifier object
    xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)

    # Create a StratifiedKFold object
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Create the RandomizedSearchCV object
    random_search = RandomizedSearchCV(estimator=xgb,
                                       param_distributions=param_dist,
                                       n_iter=250, #  Number of parameter settings that are sampled
                                    #    n_iter=3,   
                                       scoring='roc_auc',  # You can change this to the metric you want to optimize
                                    #    cv=5,  # Cross-validation splitting strategy
                                       cv=skf,
                                       verbose=0, 
                                       random_state=42)

    # Perform the randomized search
    random_search.fit(trail_X_train, trail_y_train)

    # Get the best model
    best_model = random_search.best_estimator_

    # Save the model
    trail_models[trail] = best_model

    # Get predictions for the validation set
    predictions_val = best_model.predict_proba(trail_X_val)[:, 1]

    # Calculate the ROC AUC of the predictions
    roc_auc_val = roc_auc_score(trail_y_val, predictions_val)
    
    # Save the ROC AUC score and the trail in the model evaluations
    model_evaluations.append({
        'trail': trail,
        'roc_auc_val': roc_auc_val,
    })

    # Get feature importances
    feature_importances = pd.DataFrame(best_model.feature_importances_,
                                       index=trail_X_train.columns,
                                       columns=['importance']).sort_values('importance', ascending=False)
    all_feature_importances.append(feature_importances)

    # Use SHAP to explain the model's predictions
    plt.figure(figsize=(8, 3.5))
    
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(trail_X_train)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.set_title(f"Shapley Values for {trail} Trails")
    shap.summary_plot(shap_values, trail_X_train, plot_size=None, show=False)
    plt.tick_params(axis='y', which='major', labelsize=8)  # Set y-tick label size
    plt.tight_layout()
    plt.show(block=False)

    shap_values_df = pd.DataFrame(shap_values, columns=trail_X_train.columns)
    shap_values_all.append(shap_values_df)

# Create final dataframes
feature_importances_df = pd.concat(all_feature_importances, axis=0, ignore_index=True)
shap_values_df_all = pd.concat(shap_values_all, axis=0, ignore_index=True)
model_evaluations_df = pd.DataFrame(model_evaluations)

# Initialize a list to store test set evaluations
test_evaluations = []

# Evaluate each model on the test set
for trail, model in trail_models.items():
    # Prepare test data for the specific trail
    trail_mask_test = features_test["trail"] == trail
    trail_X_test = features_test.loc[trail_mask_test, :].drop(columns=['trail'])
    trail_y_test = target_test.loc[trail_mask_test]

    # Check if there are test samples for the trail
    if len(trail_y_test) == 0:
        print(f"No test samples for trail: {trail}")
        continue
    
    # Get predictions for the test set
    predictions_test = model.predict_proba(trail_X_test)[:, 1]

    # Calculate the ROC AUC of the predictions
    roc_auc_test = roc_auc_score(trail_y_test, predictions_test)

    test_evaluations.append({
        'trail': trail,
        'roc_auc_test': roc_auc_test,
    })

# Create final dataframe
test_evaluations_df = pd.DataFrame(test_evaluations)


# %%
print(model_evaluations_df.merge(test_evaluations_df, how = 'inner', on = 'trail').sort_values('roc_auc_val', ascending=False))


# Save the dictionary of models
joblib.dump(trail_models, 'data/02_trail_models.joblib')

print("saved models")


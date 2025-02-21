# %% import libraries
import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import random  
from datetime import datetime

from pathlib import Path
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import random  # Import for random sampling
import time
# Marquer le début du calcul
start_time = time.time()

# %% fonctions
# loading dataframe, feature selection and process fold
def process_fold(x):
    # loading dataframe 
    path = Path('../inputs/data_RLS/Galaxy117-Sort_on_data_82_n_vec_subset_' + str(x) + '.csv')
    df_subset = pd.read_csv(path, sep=',', low_memory=False)
    df = df_init.merge(df_subset[['SurveyID', 'subset']], on='SurveyID', suffixes=('', '_new'), how='left')
    df['subset'] = df['subset_new'].fillna(df['subset'])
    df = df.drop(columns=['subset_new'])
    # Feature selection. The features that will be used by the model
    quantitative_col = [
    'SiteLat',
    'SiteLong',
    'B08_sentinel_band_0_central_value',
    'B08_sentinel_band_0_sd',
    'bathy_95m_band_0_mean_15x15',
    'bathy_95m_band_0_sd',
    'chlorophyll_concentration_1km_band_0_mean_15x15',
    'chlorophyll_concentration_1km_band_0_sd',
    'east_water_velocity_4_2km_mean_day_lite_band_0_mean_7x7',
    'east_water_velocity_4_2km_mean_day_lite_band_0_sd',
    'east_water_velocity_4_2km_mean_day_lite_band_1_mean_7x7',
    'east_water_velocity_4_2km_mean_day_lite_band_1_sd',
    'east_water_velocity_4_2km_mean_day_lite_band_2_mean_7x7',
    'east_water_velocity_4_2km_mean_day_lite_band_2_sd',
    'east_water_velocity_4_2km_mean_month_lite_band_0_mean_7x7',
    'east_water_velocity_4_2km_mean_month_lite_band_0_sd',
    'east_water_velocity_4_2km_mean_month_lite_band_1_mean_7x7',
    'east_water_velocity_4_2km_mean_month_lite_band_1_sd',
    'east_water_velocity_4_2km_mean_month_lite_band_2_mean_7x7',
    'east_water_velocity_4_2km_mean_month_lite_band_2_sd',
    'meditereanean_sst_band_0_mean_15x15',
    'meditereanean_sst_band_0_sd',
    'north_water_velocity_4_2km_mean_day_lite_band_0_mean_7x7',
    'north_water_velocity_4_2km_mean_day_lite_band_0_sd',
    'north_water_velocity_4_2km_mean_day_lite_band_1_mean_7x7',
    'north_water_velocity_4_2km_mean_day_lite_band_1_sd',
    'north_water_velocity_4_2km_mean_day_lite_band_2_mean_7x7',
    'north_water_velocity_4_2km_mean_day_lite_band_2_sd',
    'north_water_velocity_4_2km_mean_month_lite_band_0_mean_7x7',
    'north_water_velocity_4_2km_mean_month_lite_band_0_sd',
    'north_water_velocity_4_2km_mean_month_lite_band_1_mean_7x7',
    'north_water_velocity_4_2km_mean_month_lite_band_1_sd',
    'north_water_velocity_4_2km_mean_month_lite_band_2_mean_7x7',
    'north_water_velocity_4_2km_mean_month_lite_band_2_sd',
    'salinity_4_2km_mean_day_lite_band_0_mean_7x7',
    'salinity_4_2km_mean_day_lite_band_0_sd',
    'salinity_4_2km_mean_day_lite_band_1_mean_7x7',
    'salinity_4_2km_mean_day_lite_band_1_sd',
    'salinity_4_2km_mean_day_lite_band_2_mean_7x7',
    'salinity_4_2km_mean_day_lite_band_2_sd',
    'salinity_4_2km_mean_month_lite_band_0_mean_7x7',
    'salinity_4_2km_mean_month_lite_band_0_sd',
    'salinity_4_2km_mean_month_lite_band_1_mean_7x7',
    'salinity_4_2km_mean_month_lite_band_1_sd',
    'salinity_4_2km_mean_month_lite_band_2_mean_7x7',
    'sea_water_potential_temperature_at_sea_floor_4_2km_mean_day_band_0_mean_7x7',
    'sea_water_potential_temperature_at_sea_floor_4_2km_mean_day_band_0_sd',
    'sea_water_potential_temperature_at_sea_floor_4_2km_mean_month_band_0_mean_7x7',
    'sea_water_potential_temperature_at_sea_floor_4_2km_mean_month_band_0_sd',
    'substrate_band_0_central_value',
    'substrate_band_0_sd',
    'TCI_sentinel_band_0_central_value',
    'TCI_sentinel_band_0_sd',
    'TCI_sentinel_band_1_central_value',
    'TCI_sentinel_band_1_sd',
    'TCI_sentinel_band_2_central_value',
    'TCI_sentinel_band_2_sd',
    ]
    # load ResNet predictions
    if hybride == 'on':
        # load ResNet predictions
        df_pred_ResNet = pd.read_csv(ResNet_dir + "pred_flod_" + str(x) + '.csv', sep=',', low_memory=False)
        df_pred_ResNet = df_pred_ResNet[df_pred_ResNet['type'].str.contains('pred')]
        df_pred_ResNet[df_pred_ResNet.columns[3:]] = df_pred_ResNet.iloc[:, 3:].applymap(lambda x: max(x, 0))
        df = df.merge(df_pred_ResNet.iloc[:, [0] + list(range(3, df_pred_ResNet.shape[1]))], on='SurveyID', how='left')
        # Feature selection. The features that will be used by the model
        list_sp_pred = list(df_pred_ResNet.iloc[:, 3:].columns)
        variable_RF = quantitative_col + list_sp_pred
    else:
        variable_RF = quantitative_col
    # shuffle dataframe
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    # prepare dataset to train model
    X = df[variable_RF]
    y = [eval(item) for item in df['species_abundance_vec']]
    y = np.array(y)
    y = np.log1p(y)
    X_train = X[df.subset == 'train']
    y_train = y[df.subset == 'train']
    X_val = X[df.subset == 'val']
    y_val = y[df.subset == 'val']
    X_test = X[df.subset == 'test']
    y_test = y[df.subset == 'test']
    # return dataset to train model
    return X_train, y_train, X_val, y_val

# %% parameters
hybride = 'off'
dir_out = 'RF_abundace_optimisation'
index = 'SurveyID'
df_init = pd.read_csv('../inputs/data_RF_RSL/Galaxy117-Sort_on_data_82_n_vec_value.csv', sep=',', low_memory=False)
ResNet_dir = '../inputs/outputs_cnn_sdm_abundance_with_tl/'        
if not os.path.exists('../outputs/' + dir_out):
    os.makedirs('../outputs/' + dir_out)
random_seed = 42

# %% Optimize the hyperparameters of the Random Forest
# Set the seed for reproducibility
random.seed(random_seed)  
# Initialize dictionaries to store the results
X_train_dict = {}
y_train_dict = {}
X_val_dict = {}
y_val_dict = {}
# Process folds for values of x from 0 to 19 and store the results in the dictionaries
# Example usage: X_train_dict[0] returns the X_train of process_fold(0)
for x in range(20):
    X_train, y_train, X_val, y_val = process_fold(x)
    X_train_dict[x] = X_train
    y_train_dict[x] = y_train
    X_val_dict[x] = X_val
    y_val_dict[x] = y_val
# Define the Random Forest model
rf = RandomForestRegressor(random_state=random_seed, max_depth=None, criterion="absolute_error", n_jobs=-1)
# Define the parameter grid for GridSearchCV
n_features = len(X_train_dict[0].columns)
param_grid = [{'n_estimators': [50, 100, 200, 300, 400, 500],
               'min_samples_split': [2, 5, 10],
               'max_samples': [0.5, 0.75, 1.0],
               'bootstrap': [True],
               'max_features': [1, 'sqrt', int(n_features / 3)]},
              {'n_estimators': [50, 100, 200, 300, 400, 500],
               'min_samples_split': [2, 5, 10],
               'bootstrap': [False],
               'max_features': [1, 'sqrt', int(n_features / 3)]}]
# Define the folds to be used
folds = random.sample(range(20), 5) # Randomly sample x folds
# Initialize GridSearchCV with pre-defined train and validation sets
cv_splits = [(list(range(len(X_train_dict[fold]))),
             list(range(len(X_train_dict[fold]),
             len(X_train_dict[fold]) + len(X_val_dict[fold]))))
            for fold in folds]
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=4, cv=cv_splits)
# Combine train and validation sets for GridSearchCV
X_combined = pd.concat([pd.concat([X_train_dict[fold], X_val_dict[fold]]) for fold in folds])
y_combined = np.concatenate([np.concatenate([y_train_dict[fold], y_val_dict[fold]]) for fold in folds])
# Fit GridSearchCV
grid_search.fit(X_combined, y_combined)
# Print the best hyperparameters
print(f"Best hyperparameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_}")
# Convert grid_search.cv_results_ to a DataFrame
cv_results_df = pd.DataFrame(grid_search.cv_results_)
# Save the DataFrame to a CSV file
now = datetime.now()
timestamp = now.strftime("%Y_%m_%d_%H_%M_%S")
cv_results_df.to_csv(f'../outputs/{dir_out}/grid_search_results_{timestamp}.csv', index=False)
# Marquer la fin du calcul
end_time = time.time()
# Calculer le temps écoulé
elapsed_time = end_time - start_time
print(f"Temps d'exécution : {elapsed_time:.4f} secondes")


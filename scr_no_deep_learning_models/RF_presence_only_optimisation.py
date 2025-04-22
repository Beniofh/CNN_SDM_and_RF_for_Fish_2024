"""
Train a random forest to obtain a baseline 
top-k Macro and micro average.
The model is a random forest.
"""
# %% import libraries
import os
import pandas as pd
import random  
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, log_loss
import time
# Marquer le début du calcul
start_time = time.time()

# %% parameters
dir_out = 'RF_presence_only_optimisation'
if not os.path.exists('../outputs/' + dir_out):
    os.makedirs('../outputs/' + dir_out)
# Set the seed for reproducibility
random_seed = 42
random.seed(random_seed)  

# %% loading dataframe
# récupération du datafarme avec toutes les variables écologiques pour TOUTES les occurences
df_value = pd.read_csv('../inputs/data_RF_Gbif/Fish-refined_subset_value.csv', sep=',', low_memory=False)
# récupération du datafarme avec uniquement les occurences valides ET les bons subsets (train, val, test)
df_clean = pd.read_csv('../inputs/data_RF_Gbif/Fish-refined_clean_subset.csv', sep=',', low_memory=False)
# création d'un datafarme avec toutes les variables qui ne conserve (1) que les occurences valides et (2) qui à les bons subsets (train, val, test)
df = df_value[df_value.id.isin(df_clean.id)]
df = df.drop(['subset'], axis=1)
df = pd.merge(df, df_clean.loc[:, ['id', 'subset']], on='id')


# %% Feature selection. The features that will be used by the model
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
    'salinity_4_2km_mean_month_lite_band_2_sd',

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
qualitative_col = [
    #'ecoregion'
]

# %% processing labels.
# The labels will be consecutive numbers between 0 and the number of species
le = LabelEncoder()
le.fit(df.species.unique())
df['labels'] = le.transform(df.species).astype(int)
n_lable=len(df['labels'].unique())
# %% Processing NaN on selected features
# This step can be enhanced. Here NaN are replace by the average
# of the column values that are not NaN
for col in quantitative_col:
    df.loc[df[col].isna(), col] = df[~df[col].isna()][col].astype(float).mean()

# %% processing qualitive columns
# the qualitative variables will be replaced by a vector 
# which size is the number of modality.. All dimension will be set
# at 0 except the one corresponding to the actual current eco region
ohe = OneHotEncoder()
ohe.fit(df[['ecoregion']])
X_qual = ohe.transform(df[['ecoregion']])
onehot = pd.DataFrame(X_qual.todense())
onehot.columns = [f'eco_{i}' for i in onehot.columns]
df = pd.concat([df, onehot], axis=1)

# %% prepare dataset to train model
# the column subset contains values train, val and test
# we will test the model on train and val.
X = df[quantitative_col]
y = df.labels
X_train = X[df.subset=='train']
y_train = y[df.subset=='train']
X_val = X[df.subset=='val']
y_val = y[df.subset=='val']
X_test = X[df.subset=='test']
y_test = y[df.subset=='test']

# %% test random forest
rf = RandomForestClassifier(random_state=random_seed, max_depth=None, criterion='log_loss', n_jobs=-1)

# Define the parameter grid for GridSearchCV
n_features = len(X_train.columns)

param_grid = [{'n_estimators': [50, 100, 200, 300, 400, 500],
               'min_samples_split': [2, 5, 10],
               'max_samples': [0.5, 0.75, 1.0],
               'bootstrap': [True],
               'max_features': [1, 'sqrt', int(n_features / 3)]},
              {'n_estimators': [50, 100, 200, 300, 400, 500],
               'min_samples_split': [2, 5, 10],
               'bootstrap': [False],
               'max_features': [1, 'sqrt', int(n_features / 3)]}]

# rechercher les classes uniques dans y_train qui ne sont pas présente dana y_val
all_classes = np.unique(np.concatenate([y_train, y_val]))

# Combine train and validation sets for GridSearchCV
X_combined = pd.concat([X_train, X_val])
y_combined = np.concatenate([y_train, y_val])

# Initialize GridSearchCV with pre-defined train and validation sets
cv_splits = [(list(range(len(X_train))), list(range(len(X_train), len(X_train) + len(X_val))))]
grid_search = GridSearchCV(estimator=rf,
                           param_grid=param_grid,
                           scoring='f1_macro',
                           n_jobs=2,
                           verbose=4,
                           cv=cv_splits)

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

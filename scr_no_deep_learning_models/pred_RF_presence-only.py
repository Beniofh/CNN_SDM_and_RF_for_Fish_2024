"""
Train a random forest to obtain a baseline 
top-k Macro and micro average.
The model is a random forest.
"""
# %% import libraries
import os
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import top_k_accuracy_score
from sklearn.inspection import permutation_importance

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

    #'B08_sentinel_band_0_mean_300x300',
    #'B08_sentinel_band_0_mean_149x149',
    'B08_sentinel_band_0_central_value',
    'B08_sentinel_band_0_sd',
    
    #'bathymetry_band_0_mean_9x9',
    #'bathymetry_band_0_sd',
    
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
    #'east_water_velocity_4_2km_mean_year_lite_band_0_mean_7x7',
    #'east_water_velocity_4_2km_mean_year_lite_band_0_sd',
    #'east_water_velocity_4_2km_mean_year_lite_band_1_mean_7x7',
    #'east_water_velocity_4_2km_mean_year_lite_band_1_sd',
    #'east_water_velocity_4_2km_mean_year_lite_band_2_mean_7x7',
    #'east_water_velocity_4_2km_mean_year_lite_band_2_sd',

    #'mpa_band_0_mean_256x256',
    #'mpa_band_0_mean_127x127',
    #'mpa_band_0_central_value',
    #'mpa_band_0_sd',

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
    #'north_water_velocity_4_2km_mean_year_lite_band_0_mean_7x7',
    #'north_water_velocity_4_2km_mean_year_lite_band_0_sd',
    #'north_water_velocity_4_2km_mean_year_lite_band_1_mean_7x7',
    #'north_water_velocity_4_2km_mean_year_lite_band_1_sd',
    #'north_water_velocity_4_2km_mean_year_lite_band_2_mean_7x7',
    #'north_water_velocity_4_2km_mean_year_lite_band_2_sd',

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
    #'salinity_4_2km_mean_year_lite_band_0_mean_7x7',
    #'salinity_4_2km_mean_year_lite_band_0_sd',
    #'salinity_4_2km_mean_year_lite_band_1_mean_7x7',
    #'salinity_4_2km_mean_year_lite_band_1_sd',
    #'salinity_4_2km_mean_year_lite_band_2_mean_7x7',
    #'salinity_4_2km_mean_year_lite_band_2_sd',

    'sea_water_potential_temperature_at_sea_floor_4_2km_mean_day_band_0_mean_7x7',
    'sea_water_potential_temperature_at_sea_floor_4_2km_mean_day_band_0_sd',
    'sea_water_potential_temperature_at_sea_floor_4_2km_mean_month_band_0_mean_7x7',
    'sea_water_potential_temperature_at_sea_floor_4_2km_mean_month_band_0_sd',
    #'sea_water_potential_temperature_at_sea_floor_4_2km_mean_year_band_0_mean_7x7',
    #'sea_water_potential_temperature_at_sea_floor_4_2km_mean_year_band_0_sd',

	
    #'substrate_band_0_mean_256x256',
    #'substrate_band_0_mean_127x127',
    'substrate_band_0_central_value',
    'substrate_band_0_sd',
    
    #'TCI_sentinel_band_0_mean_300x300',
    #'TCI_sentinel_band_0_mean_149x149',
    'TCI_sentinel_band_0_central_value',
    'TCI_sentinel_band_0_sd',
    #'TCI_sentinel_band_1_mean_300x300',
    #'TCI_sentinel_band_1_mean_149x149',
    'TCI_sentinel_band_1_central_value',
    'TCI_sentinel_band_1_sd',
    #'TCI_sentinel_band_2_mean_300x300',
    #'TCI_sentinel_band_2_mean_149x149',
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

#X = df[quantitative_col+onehot.columns.tolist()]
X = df[quantitative_col]
y = df.labels
X_train = X[df.subset=='train']
y_train = y[df.subset=='train']
X_val = X[df.subset=='val']
y_val = y[df.subset=='val']
X_test = X[df.subset=='test']
y_test = y[df.subset=='test']
# %% test random forest
#rf = RandomForestClassifier(max_depth=10, random_state=42)
rf = RandomForestClassifier(bootstrap=True,
                            max_features='sqrt',
                            max_samples=0.75,
                            min_samples_split=2,
                            n_estimators=200,
                            criterion="log_loss",
                            max_depth=None, 
                            n_jobs=-1,
                            random_state=42,)
rf.fit(X_train, y_train)

# %% top-k accuracy validation set
prior = df.groupby('labels')[['id']].count().sort_values('id', ascending=False)
prior.loc[:, 'id'] /= prior['id'].sum()
print('')

for i in ("test","val","train") :
    print(f'For {i} set :')
    y_temp = eval(f'y_{i}')
    X_temp = eval(f'X_{i}')

    for k in (1, 5, 10, 20):
        topk = top_k_accuracy_score(y_temp,
                                    rf.predict_proba(X_temp),
                                    k=k,
                                    labels=range(n_lable))
        print(f'(Micro avg) Top-{k} accuracy : {topk} (prior : {prior.iloc[:k]["id"].sum()})')

# Macro average top-K accuracy weights computation
# each species must contribute with the same overall weight in macro average.
# we compute the number of occurence by species. The weight of a species is
# 1/number_of_occurences. This if one species has 10 occurrences each, successfully
# predicted by the model, the accuracy contribution will be 10/10/205=1/205.
    dfw = df[df.subset == i]
    weights = dfw.groupby('labels')\
                 .count()[['id']]\
                 .apply(lambda a: 1/a).rename({'id': 'weight'}, axis=1)
    weights.columns = ['weight']
    dfw = dfw.join(weights, how='left', on='labels')

    # compute weighted top-K validation set
    for k in (1, 5, 10, 20):
        topk = top_k_accuracy_score(y_temp,
                                    rf.predict_proba(X_temp),
                                    k=k,
                                    labels=range(n_lable),
                                    sample_weight=dfw.weight)
        print(f'(Macro avg) Top-{k} accuracy : {topk} (prior : {k/df.labels.nunique()})')

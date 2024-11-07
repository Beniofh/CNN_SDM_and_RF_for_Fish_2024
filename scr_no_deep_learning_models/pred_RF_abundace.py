# %% import libraries
import os
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, d2_absolute_error_score, mean_gamma_deviance, mean_poisson_deviance, explained_variance_score, max_error, mean_squared_error
from pathlib import Path
import shutil
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
from joblib import Parallel, delayed  # Import for parallelization
import time

# Marquer le début du calcul
start_time = time.time()

# %% parameters
hybride = 'off'
dir_out = 'pred_RF_abundace'
index = 'SurveyID'
df_init = pd.read_csv('../inputs/data_RF_RSL/Galaxy117-Sort_on_data_82_n_vec_value.csv', sep=',', low_memory=False)
ResNet_dir = '../inputs/outputs_cnn_sdm_abundance_with_tl/'        

if not os.path.exists('../outputs/' + dir_out):
    os.makedirs('../outputs/' + dir_out)

# Function to process each fold
def process_fold(x):
    # %% loading dataframe 
    path = Path('../inputs/data_RLS/Galaxy117-Sort_on_data_82_n_vec_subset_' + str(x) + '.csv')
    
    df_subset = pd.read_csv(path, sep=',', low_memory=False)
    df = df_init.merge(df_subset[['SurveyID', 'subset']], on='SurveyID', suffixes=('', '_new'), how='left')
    df['subset'] = df['subset_new'].fillna(df['subset'])
    df = df.drop(columns=['subset_new'])

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
    # %% load ResNet predictions
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

    # %% shuffle dataframe
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # %% prepare dataset to train model
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

    # %% test random forest
    rf = RandomForestRegressor(max_depth=10, criterion="absolute_error")
    rf.fit(X_train, y_train)

   
    # %% predict
    list_sp = ['Apogon imberbis', 'Atherina hepsetus', 'Boops boops', 'Centrolabrus exoletus', 'Chelon labrosus', 'Chromis chromis', 'Chrysophrys auratus', 'Coris julis', 'Ctenolabrus rupestris', 'Dentex dentex', 'Diplodus annularis', 'Diplodus cervinus', 'Diplodus puntazzo', 'Diplodus sargus', 'Diplodus vulgaris', 'Epinephelus costae', 'Epinephelus marginatus', 'Gobius bucchichi', 'Gobius xanthocephalus', 'Labrus merula', 'Labrus viridis', 'Mullus surmuletus', 'Muraena helena', 'Oblada melanura', 'Octopus vulgaris', 'Pagrus pagrus', 'Parablennius pilicornis', 'Parablennius rouxi', 'Pomadasys incisus', 'Sarpa salpa', 'Sciaena umbra', 'Seriola dumerili', 'Serranus cabrilla', 'Serranus scriba', 'Spicara maena', 'Spicara smaris', 'Spondyliosoma cantharus', 'Symphodus cinereus', 'Symphodus doderleini', 'Symphodus mediterraneus', 'Symphodus melanocercus', 'Symphodus ocellatus', 'Symphodus roissali', 'Symphodus rostratus', 'Symphodus tinca', 'Thalassoma pavo', 'Tripterygion delaisi']
    
    id_and_subset = pd.concat([df[['SurveyID','subset']][df.subset=='train'],
                           df[['SurveyID','subset']][df.subset=='val'],
                           df[['SurveyID','subset']][df.subset=='test']
                          ],ignore_index=True)
    
    df_pred = pd.concat([pd.DataFrame(rf.predict(X_train)),
                         pd.DataFrame(rf.predict(X_val)),
                         pd.DataFrame(rf.predict(X_test))
                        ],ignore_index=True)
    df_pred[df_pred < 0] = 0
    df_pred = df_pred.rename(columns=dict(zip(df_pred.columns, list_sp)))
    df_pred = pd.concat([id_and_subset, df_pred], axis=1)
    df_pred.insert(2, "type", ['pred_{}'.format(i+1) for i in range(len(df_pred))])
    

    df_true = pd.concat([pd.DataFrame(y_train),
                         pd.DataFrame(y_val),
                         pd.DataFrame(y_test)
                        ],ignore_index=True)
    df_true = df_true.rename(columns=dict(zip(df_true.columns, list_sp)))  
    df_true = pd.concat([id_and_subset, df_true], axis=1)
    df_true.insert(2, "type", ['true_{}'.format(i+1) for i in range(len(df_true))])
    
    df_final = pd.concat([df_pred, df_true], ignore_index=True)

    df_pred_test_stack = pd.DataFrame(df_pred.iloc[:, 3:][df_pred["subset"]=="test"].stack().reset_index(drop=True))
    df_pred_test_stack.insert(0, "subset", "test")
    df_pred_test_stack.insert(0, "specie", list_sp*df_pred[df_pred["subset"]=="test"].shape[0])
    df_pred_test_stack.insert(0, "SurveyID", [item for item in df_pred["SurveyID"][df_pred["subset"]=="test"].tolist() for _ in range(len(list_sp))])

    df_pred_val_stack = pd.DataFrame(df_pred.iloc[:, 3:][df_pred["subset"]=="val"].stack().reset_index(drop=True))
    df_pred_val_stack.insert(0, "subset", "val")
    df_pred_val_stack.insert(0, "specie", list_sp*df_pred[df_pred["subset"]=="val"].shape[0])
    df_pred_val_stack.insert(0, "SurveyID", [item for item in df_pred["SurveyID"][df_pred["subset"]=="val"].tolist() for _ in range(len(list_sp))])

    df_pred_train_stack = pd.DataFrame(df_pred.iloc[:, 3:][df_pred["subset"]=="train"].stack().reset_index(drop=True))
    df_pred_train_stack.insert(0, "subset", "train")
    df_pred_train_stack.insert(0, "specie", list_sp*df_pred[df_pred["subset"]=="train"].shape[0])
    df_pred_train_stack.insert(0, "SurveyID", [item for item in df_pred["SurveyID"][df_pred["subset"]=="train"].tolist() for _ in range(len(list_sp))])

    df_true_test_stack = df_true.iloc[:, 3:][df_true["subset"]=="test"].stack().reset_index(drop=True)
    df_true_val_stack = df_true.iloc[:, 3:][df_true["subset"]=="val"].stack().reset_index(drop=True)
    df_true_train_stack = df_true.iloc[:, 3:][df_true["subset"]=="train"].stack().reset_index(drop=True)

    df_final_stack = pd.concat(
        [pd.concat([df_pred_test_stack, df_pred_val_stack, df_pred_train_stack], ignore_index=True),
        pd.concat([df_true_test_stack, df_true_val_stack, df_true_train_stack], ignore_index=True)],
        axis=1)
   
    df_final_stack.columns=["SurveyID", "specie", "subset","pred", "true"]
        
    # %% save predictions
    df_final.to_csv('../outputs/' + dir_out + '/test_pred_flod_' + str(x) + '.csv', index=False)
    df_final_stack.to_csv('../outputs/' + dir_out + '/test_pred_flod_' + str(x) + '_stacked.csv', index=False)

    # %% compute and save Mean Decrease in Impurity graph
    mdi_importances = pd.Series(
        rf.feature_importances_,
        index=rf.feature_names_in_
    ).sort_values(ascending=True)

    figure, ax = plt.subplots(figsize=(20, 6))
    mdi_importances[-15:].plot.barh(ax=ax)
    ax.set_title("Random Forest Feature Importances")
    ax.set_xlabel("Mean Decrease in Impurity (MDI)")
    ax.set_xlim(0, 0.1)
    figure.tight_layout()
    figure.savefig('../outputs/' + dir_out + '/Mean_Decrease_in_Impurity_flod_' + str(x) + '.png', bbox_inches='tight')

    # %% compute and save Permutation Importances graph
    result = permutation_importance(
        rf, X_val, y_val)
    sorted_importances_idx = result.importances_mean.argsort()
    importances = pd.DataFrame(
        result.importances[sorted_importances_idx].T,
        columns=X.columns[sorted_importances_idx],
    )
    figure, ax = plt.subplots(figsize=(20, 6))
    importances.iloc[:, -20:].plot.box(vert=False, whis=10, ax=ax)
    ax.set_title("Permutation Importances (val set)")
    ax.axvline(x=0, color="k", linestyle="--")
    ax.set_xlabel("Decrease in accuracy score")
    figure.tight_layout()
    figure.savefig('../outputs/' + dir_out + '/Permutation_Importances_flod_' + str(x) + '.png', bbox_inches='tight')

    print(f"Fold {x} processed")

    return x  # Return fold index to track progress

# Save the current script
current_script_path = os.path.abspath(__file__)
shutil.copy2(current_script_path, '../outputs/' + dir_out + '/config.py')

# Run the folds in parallel with progress tracking
n_jobs = -1  # Use all available cores
results = Parallel(n_jobs=n_jobs)(
    delayed(process_fold)(x) for x in tqdm(range(0, 20), desc="Processing Random Forest Folds")
)
# Marquer la fin du calcul
end_time = time.time()

# Calculer le temps écoulé
elapsed_time = end_time - start_time
print(f"Temps d'exécution : {elapsed_time:.4f} secondes")
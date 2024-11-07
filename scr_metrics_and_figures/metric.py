import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, explained_variance_score, max_error, mean_squared_error, d2_absolute_error_score, median_absolute_error
from scipy.stats import spearmanr, pearsonr

# Dossiers et fichiers

data_sets = {
    'CNN_SDM_with_tl': {
        'files': [f'pred_flod_{i}' for i in range(20)], 
        'path': '../outputs/pred_cnn_sdm_abundance_with_tl/{}.csv'
    },
    'CNN_SDM_without_tl': {
        'files': [f'pred_flod_{i}' for i in range(20)], 
        'path': '../outputs/pred_cnn_sdm_abundance_without_tl/{}.csv'
    },
    'RF': {
        'files': [f'pred_flod_{i}' for i in range(20)],
        'path': '../inputs/output_RF_abundace/{}.csv'
    },

}

# Fonction pour calculer la médiane, le nombre d'occurrences non nulles, la somme totale, le minimum et le maximum
def stats(series):
    # Filtrer les valeurs non nulles
    non_zero_values = series[series != 0]
    # Calculer les statistiques
    median_value = non_zero_values.median()
    count_non_zero = non_zero_values.count()
    total_sum = series.sum()
    min_value = non_zero_values.min() if not non_zero_values.empty else None
    max_value = non_zero_values.max() if not non_zero_values.empty else None
    std_value = series.std() if not non_zero_values.empty else None

    return pd.Series({
        'mediane': median_value,
        'occurrences': count_non_zero,
        'sum': total_sum,
        'min': min_value,
        'max': max_value,
        'std': std_value,
    })

# Fonctions pour calculer les métriques
def compute_metrics(df_pred, df_true):
    test_pred = df_pred.values
    test_true = df_true.values
    test_pred_t = test_pred.T
    test_true_t = test_true.T
    test_pred_flat = test_pred.flatten()
    test_true_flat = test_true.flatten()

    spearman_coefficients_site = []
    for x in range(test_true.shape[0]):
        coefficient_site, _ = spearmanr(test_true[x], test_pred[x])
        spearman_coefficients_site.append(coefficient_site)

    return {
        #'mean_absolute_log_error': mean_absolute_error(test_true, test_pred),
        #'mean_R2_Site_log': r2_score(test_true_t, test_pred_t),
        #'mean_R2_Site_log_vw': r2_score(test_true_t, test_pred_t, multioutput='variance_weighted'),
        #'mean_R2_Species_log': r2_score(test_true, test_pred),
        #'mean_R2_Species_log_vw': r2_score(test_true, test_pred, multioutput='variance_weighted'),
        'R2_log_score': r2_score(test_true_flat, test_pred_flat),
        #'mean_R2_log_vw': r2_score(test_true_flat, test_pred_flat, multioutput='variance_weighted'),
        #'explained_variance_log_score': explained_variance_score(test_true_flat, test_pred_flat),
        #'max_log_error': max_error(test_true_flat, test_pred_flat),
        #'mean_squared_log_error': mean_squared_error(test_true, test_pred),
        'd2_absolute_log_error_score': d2_absolute_error_score(test_true_flat, test_pred_flat),
        #'d2_absolute_log_error_score_by_site': d2_absolute_error_score(test_true_t, test_pred_t),
        'spearman_coef': spearmanr(test_true_flat, test_pred_flat)[0],
        #'spearmanr_coef_site' : np.mean(spearman_coefficients_site),
        #'r_regression' : pearsonr(test_true_flat, test_pred_flat)[0],
        #'median_A_E' : median_absolute_error(test_true, test_pred),
        #'SPE': np.mean((test_true - test_pred)/np.var(test_true)),
    }

def compute_metrics_by_sp(df_pred, df_true):
    test_pred = df_pred.values
    test_true = df_true.values
    test_pred_t = test_pred.T
    test_true_t = test_true.T
    test_pred_t_round = np.log(((np.exp(test_pred_t)-1).round(0)) + 1).round(2)
    test_true_t_round = np.log(((np.exp(test_true_t)-1).round(0)) + 1).round(2)
    spearman_coefficients = []
    underestimate = []
    overestimate = []
    sum_error = []

    for x in range(test_true.shape[1]):
        #coefficient, _ = spearmanr(test_true[x], test_pred[x])
        coefficient, _ = spearmanr(test_true_t[x], test_pred_t[x])
        spearman_coefficients.append(coefficient)
        underestimate_temp = test_pred_t_round[x] < test_true_t_round[x]
        underestimate.append(underestimate_temp.sum().sum())
        overestimate_temp = test_pred_t_round[x] > test_true_t_round[x]
        overestimate.append(overestimate_temp.sum().sum())
        sum_error.append(overestimate_temp.sum().sum()+underestimate_temp.sum().sum())

    return {
        #'R2_Species_log': r2_score(test_true, test_pred, multioutput='raw_values'),
        'd2_absolute_log_error_score_by_sp': d2_absolute_error_score(test_true, test_pred, multioutput='raw_values'),
        #'mean_absolute_log_error_sp' : mean_absolute_error(test_true, test_pred, multioutput='raw_values'),
        #'spearmanr_coef': spearman_coefficients,
        #'underestimate_by_sp': underestimate,
        #'overestimate_by_sp': overestimate,
        #'sum_error_by_sp': sum_error
    }

# Renommer les colonnes avec les stats
path_dir = "../inputs/data_RF_RSL/"
data = "Galaxy117-Sort_on_data_82_n"
df = pd.read_csv(path_dir + "/" + data + ".csv", sep=',')
df = df.iloc[:, 9:]
results = df.apply(stats)
sp_stats = results.loc['occurrences'].tolist()

# Calcul des métriques pour chaque ensemble de données
for key, value in data_sets.items():
    value['metrics'] = []
    value['metrics_by_sp'] = []
    for f in value['files']:
        df = pd.read_csv(value['path'].format(f))
        '''
        if df.columns[0] == 'SurveyID':    
            filtered_columns = [col for col, occ in zip(df.iloc[:, 3:].columns, sp_stats) if occ < 45]
        else:
            filtered_columns = [col for col, occ in zip(df.iloc[:, 1:].columns, sp_stats) if occ < 45]
        '''
        '''        
        df.drop(['Coris julis',
                'Chromis chromis', 
                'Diplodus vulgaris',
                'Symphodus tinca',
                'Thalassoma pavo',
                'Serranus scriba',
                'Diplodus sargus',
                'Serranus cabrilla',
                'Mullus surmuletus',
                'Symphodus ocellatus',
                'Symphodus mediterraneus',
                'Sarpa salpa',
                'Symphodus roissali',
                'Oblada melanura',
                'Diplodus annularis',
                'Boops boops',
                'Apogon imberbis',
                'Symphodus rostratus',
                'Parablennius pilicornis',
                'Diplodus puntazzo',
                'Spondyliosoma cantharus',
                'Labrus merula',
                'Symphodus melanocercus',
                'Epinephelus marginatus',
                'Spicara maena',
                'Tripterygion delaisi',
                'Diplodus cervinus',
                'Gobius bucchichi',
                'Symphodus doderleini',
                'Epinephelus costae',
                ], axis=1, inplace=True)
        '''
        '''
        df.drop(['Boops boops',
                'Atherina hepsetus',
                'Spicara maena',
                'Spicara smaris',                
                ], axis=1, inplace=True)
        '''
        '''
        df.drop(['Ctenolabrus rupestris',
                'Chelon labrosus',
                'Labrus viridis',
                'Muraena helena',
                'Atherina hepsetus',
                'Octopus vulgaris',
                'Pagrus pagrus',
                'Spicara smaris',
                'Centrolabrus exoletus',
                'Gobius xanthocephalus',
                'Sciaena umbra',
                'Symphodus cinereus',
                'Chrysophrys auratus',
                'Parablennius rouxi',
                'Seriola dumerili',
                'Dentex dentex',
                'Pomadasys incisus',               
                ], axis=1, inplace=True)
        '''
        df = df[df["subset"]=="test"]
        df = df.iloc[:, 2:]
        df_pred = df[df['type'].str.contains('pred')].iloc[:, 1:]
        df_true = df[df['type'].str.contains('true')].iloc[:, 1:]
        df_pred[df_pred < 0] = 0  # Correction spécifique pour les ResNet

        #df_pred = np.log(((np.exp(df_pred)-1).round(0)) + 1).round(2)
        #df_true = df_true.round(2)
        
        value['metrics'].append(compute_metrics(df_pred, df_true))
        value['metrics_by_sp'].append(compute_metrics_by_sp(df_pred, df_true))

# Récupération des noms de colonnes des métriques
metric_names = list(data_sets['RF']['metrics'][0].keys())
metric_names_by_sp = list(data_sets['RF']['metrics_by_sp'][0].keys())

# Création du DataFrame des métriques
data = {
    'fold': range(len(data_sets['RF']['files']))
}

for model_key in data_sets:
    for metric_name in metric_names:
        data[f'{model_key}_{metric_name}'] = [
            m[metric_name] for m in data_sets[model_key]['metrics']
        ]



data_by_site= pd.DataFrame(columns=['model', 'metric', 'fold'] + list(df.iloc[:, 1:].columns))
for metric_name_by_sp in metric_names_by_sp:
    for model_key in data_sets:
        for i in range(len(data_sets['RF']['files'])):
            data_by_site.loc[len(data_by_site)] = [model_key, metric_name_by_sp, i] + list(data_sets[model_key]['metrics_by_sp'][i][metric_name_by_sp])

df_metrics = pd.DataFrame(data)
df_metrics.to_csv(f'../outputs/abundances_results/mectric.csv', index=False)
data_by_site.to_csv(f'../outputs/abundances_results/mectric_by_sp.csv', index=False)
import pandas as pd
import numpy as np
import os
from scipy.stats import kruskal, shapiro, levene
from scikit_posthocs import posthoc_dunn
from scipy.stats import f_oneway
from scipy.stats import permutation_test

input_dir = '../outputs/abundances_results'
input_file = 'metric.csv'
output_dir = '../outputs/abundances_results'

df = pd.read_csv(os.path.join(input_dir, input_file))

# Création d'un nouveau DataFrame avec transformation log(x+1) sauf pour la colonne 'fold'
df_trans = df.copy()
for col in df_trans.columns:
    if col != 'fold':
        df_trans[col] = 1 / df_trans[col]


def homocedasticity_test(df, metric_name, column_names):
    # Test de Levene pour l'homocédasticité
    stat, p_value = levene(
        df[column_names[0]],
        df[column_names[1]],
        df[column_names[2]])
    
    results_df = pd.DataFrame({
        'Test': ['Levene'],
        'Statistique': [stat],
        'P-valeur': [p_value]})
    print("\nRésumé des résultats du test de Levene :")
    print(results_df)
    # enregistrer les résultats
    out_file_path = os.path.join(output_dir, f'stat_test_levene_{metric_name}.csv')
    results_df.to_csv(out_file_path, index=False)
    print("Résultats du test de Levene enregistrés dans", out_file_path)

def normality_test(df, metric_name, column_names):
    # Test de Shapiro-Wilk pour la normalité
    results = []
    for col in column_names:
        stat, p_value = shapiro(df[col])
        results.append({
            '': col,
            'Statistique de Shapiro-Wilk': stat,
            'P-valeur': p_value
        })
    results_df = pd.DataFrame(results)
    print("\nRésumé des résultats du test de normalité :")
    print(results_df)
    # enregistrer les résultats
    out_file_path = os.path.join(output_dir, f'stat_test_shapiro_wilk_{metric_name}.csv')
    results_df.to_csv(out_file_path, index=False)

def Welch_ANOVA(df, metric_name, column_names):
    # ANOVA de Welch
    stat, p_value = f_oneway(
        df[column_names[0]],
        df[column_names[1]],
        df[column_names[2]])
    results_df = pd.DataFrame({
        'Test': ['ANOVA de Welch'],
        'Statistique': [stat],
        'P-valeur': [p_value]})
    print("\nRésumé des résultats du test de l'ANOVA de Welch :")
    print(results_df)
    # enregistrer les résultats
    results_df.to_csv(os.path.join(output_dir, f'stat_test_ANOVA_{metric_name}.csv'), index=False)
    print("Résultats de l'ANOVA de Welch enregistrés dans", os.path.join(output_dir, f'stat_test_ANOVA_{metric_name}.csv'))

def Dunn_post_hoc(df, metric_name, column_names):
    # Effectuer le test de Dunn's post-hoc
    data = pd.melt(df[column_names])
    dunn_results = posthoc_dunn(data, val_col='value', group_col='variable', p_adjust='sidak')
    print("Résultats du test de Dunn's post-hoc :")
    print(dunn_results)
    # Enregistrer les résultats du test de Dunn's
    dunn_results.to_csv(os.path.join(output_dir, f'stat_test_Dunn_{metric_name}.csv'))
    print("Résultats du test de Dunn's enregistrés dans", os.path.join(output_dir, f'stat_test_Dunn_{metric_name}.csv'))

def permutation_tests(df, metric_name, column_names, n_permutations=50000):

    results = pd.DataFrame(index=column_names, columns=column_names, dtype=float)
    for i, col1 in enumerate(column_names):
        for j, col2 in enumerate(column_names):
            if i <= j:
                data1 = df[col1].dropna()
                data2 = df[col2].dropna()
                result = permutation_test((data1, data2), 
                                           statistic=lambda x, y: np.mean(x) - np.mean(y),
                                           permutation_type='independent',
                                           n_resamples=n_permutations,
                                           alternative='two-sided',
                                           random_state=42)
                p_value = result.pvalue
                results.loc[col1, col2] = p_value
                results.loc[col2, col1] = p_value
            #elif i == j:
            #    results.loc[col1, col2] = np.nan
    print(f"\nTableau des p-values des tests de permutation pour {metric_name} :")
    print(results)
    out_file_path = os.path.join(output_dir, f'stat_test_permutation_{metric_name}.csv')
    results.to_csv(out_file_path)
    print("Résultats des tests de permutation enregistrés dans", out_file_path)


permutation_tests(df,
                 'R2log',
                 ['CNN_SDM_with_tl_R2_log_score',
                  'CNN_SDM_without_tl_R2_log_score',
                  'RF_R2_log_score'])

permutation_tests(df,
                 'D2log',
                 ['CNN_SDM_with_tl_d2_absolute_log_error_score',
                  'CNN_SDM_without_tl_d2_absolute_log_error_score',
                  'RF_d2_absolute_log_error_score'])   

permutation_tests(df,
                 'Spearman',
                 ['CNN_SDM_with_tl_spearman_coef',
                  'CNN_SDM_without_tl_spearman_coef',
                  'RF_spearman_coef'])



homocedasticity_test(df,
                    'R2log',
                    ['CNN_SDM_with_tl_R2_log_score',
                    'CNN_SDM_without_tl_R2_log_score',
                    'RF_R2_log_score'])

homocedasticity_test(df,
                    'D2log',
                    ['CNN_SDM_with_tl_d2_absolute_log_error_score',
                    'CNN_SDM_without_tl_d2_absolute_log_error_score',
                    'RF_d2_absolute_log_error_score'])

homocedasticity_test(df_trans,
                    'D2log_transformed',
                    ['CNN_SDM_with_tl_d2_absolute_log_error_score',
                    'CNN_SDM_without_tl_d2_absolute_log_error_score',
                    'RF_d2_absolute_log_error_score'])

homocedasticity_test(df,
                    'Spearman',
                    ['CNN_SDM_with_tl_spearman_coef',
                    'CNN_SDM_without_tl_spearman_coef',
                    'RF_spearman_coef'])

normality_test(df,
               'R2log',
                ['CNN_SDM_with_tl_R2_log_score',
                 'CNN_SDM_without_tl_R2_log_score',
                 'RF_R2_log_score'])
normality_test(df,
                'D2log',
                ['CNN_SDM_with_tl_d2_absolute_log_error_score',
                 'CNN_SDM_without_tl_d2_absolute_log_error_score',
                 'RF_d2_absolute_log_error_score'])

normality_test(df_trans,
                'D2log_transformed',
                ['CNN_SDM_with_tl_d2_absolute_log_error_score',
                 'CNN_SDM_without_tl_d2_absolute_log_error_score',
                 'RF_d2_absolute_log_error_score'])
normality_test(df,
                'Spearman',
                ['CNN_SDM_with_tl_spearman_coef',
                 'CNN_SDM_without_tl_spearman_coef',
                 'RF_spearman_coef'])

Welch_ANOVA(df,
            'R2log',
            ['CNN_SDM_with_tl_R2_log_score',
             'CNN_SDM_without_tl_R2_log_score',
             'RF_R2_log_score'])

Welch_ANOVA(df,
            'D2log',
            ['CNN_SDM_with_tl_d2_absolute_log_error_score',
             'CNN_SDM_without_tl_d2_absolute_log_error_score',
             'RF_d2_absolute_log_error_score'])

Welch_ANOVA(df_trans,
            'D2log_transformed',
            ['CNN_SDM_with_tl_d2_absolute_log_error_score',
             'CNN_SDM_without_tl_d2_absolute_log_error_score',
             'RF_d2_absolute_log_error_score'])

Welch_ANOVA(df,
            'Spearman',
            ['CNN_SDM_with_tl_spearman_coef',
             'CNN_SDM_without_tl_spearman_coef',
             'RF_spearman_coef'])


Dunn_post_hoc(df,
             'R2log',
             ['CNN_SDM_with_tl_R2_log_score',
              'CNN_SDM_without_tl_R2_log_score',
              'RF_R2_log_score'])

Dunn_post_hoc(df,
             'D2log',
             ['CNN_SDM_with_tl_d2_absolute_log_error_score',
              'CNN_SDM_without_tl_d2_absolute_log_error_score',
              'RF_d2_absolute_log_error_score'])

Dunn_post_hoc(df_trans,
             'D2log_transformed',
             ['CNN_SDM_with_tl_d2_absolute_log_error_score',
              'CNN_SDM_without_tl_d2_absolute_log_error_score',
              'RF_d2_absolute_log_error_score'])

Dunn_post_hoc(df,
             'Spearman',
             ['CNN_SDM_with_tl_spearman_coef',
              'CNN_SDM_without_tl_spearman_coef',
              'RF_spearman_coef'])
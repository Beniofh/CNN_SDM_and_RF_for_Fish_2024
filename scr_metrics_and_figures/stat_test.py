import pandas as pd
import numpy as np
import os
from scipy.stats import kruskal, shapiro, levene
from scikit_posthocs import posthoc_dunn

input_dir = '../outputs/abundances_results'
input_file = 'metric.csv'
output_dir = '../outputs/abundances_results'

df = pd.read_csv(os.path.join(input_dir, input_file))


def statistical_test(df, metric_name, column_names):
    # Effectuer le test de Kruskal-Wallis
    stat, p_value = kruskal(
        df[column_names[0]],
        df[column_names[1]],
        df[column_names[2]])
    # Afficher les résultats
    print("Statistique de Kruskal-Wallis:", stat)
    print("P-valeur:", p_value)
    # Enregistrer les résultats du test de Kruskal-Wallis
    kw_results = pd.DataFrame({
        'Test': ['Kruskal-Wallis'],
        'Statistique': [stat],
        'P-valeur': [p_value]})
    kw_results.to_csv(os.path.join(output_dir, f'stat_test_KW_{metric_name}.csv'), index=False)
    print("Résultats du test de Kruskal-Wallis enregistrés dans", os.path.join(output_dir, f'stat_test_KW_{metric_name}.csv'))
    # Interprétation
    if p_value < 0.05:
        print("Les distributions des scores sont significativement différentes (p < 0.05).")
        # Effectuer le test de Dunn's post-hoc
        data = pd.melt(df[column_names])
        dunn_results = posthoc_dunn(data, val_col='value', group_col='variable', p_adjust='sidak')
        print("Résultats du test de Dunn's post-hoc :")
        print(dunn_results)
        # Enregistrer les résultats du test de Dunn's
        dunn_results.to_csv(os.path.join(output_dir, f'stat_test_Dunn_{metric_name}.csv'))
        print("Résultats du test de Dunn's enregistrés dans", os.path.join(output_dir, f'stat_test_Dunn_{metric_name}.csv'))
    else:
        print("Les distributions des scores ne sont pas significativement différentes (p >= 0.05).")


statistical_test(df,
                 'R2log',
                 ['CNN_SDM_with_tl_R2_log_score',
                  'CNN_SDM_without_tl_R2_log_score',
                  'RF_R2_log_score'])

statistical_test(df,
                 'D2log',
                 ['CNN_SDM_with_tl_d2_absolute_log_error_score',
                  'CNN_SDM_without_tl_d2_absolute_log_error_score',
                  'RF_d2_absolute_log_error_score'])

statistical_test(df,
                 'Spearman',
                 ['CNN_SDM_with_tl_spearman_coef',
                  'CNN_SDM_without_tl_spearman_coef',
                  'RF_spearman_coef'])
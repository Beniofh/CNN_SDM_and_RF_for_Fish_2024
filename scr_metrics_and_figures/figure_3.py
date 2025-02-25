import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime


# Fonction pour calculer la médiane, le nombre d'occurrences non nulles, la somme totale, le minimum et le maximum
def stats(series):
    # Filtrer les valeurs non nulles
    non_zero_values = series[series != 0]
    # Calculer les statistiques
    mean_value = non_zero_values.mean()
    median_value = non_zero_values.median()
    count_non_zero = non_zero_values.count()
    total_sum = series.sum()
    min_value = non_zero_values.min() if not non_zero_values.empty else None
    max_value = non_zero_values.max() if not non_zero_values.empty else None
    std_value = series.std() if not non_zero_values.empty else None

    return pd.Series({
        'mean': mean_value,
        'mediane': median_value,
        'occurrences': count_non_zero,
        'sum': total_sum,
        'min': min_value,
        'max': max_value,
        'std': std_value,
    })

# Charger le fichier CSV dans un DataFrame
df = pd.read_csv("../outputs/abundances_results/metric_by_sp.csv")

# Séparer les données pour deux modèles différents
df_2 = df[df['model'] == "CNN_SDM_with_tl"]
df_1 = df[df['model'] == "RF"]

# Calculer la différence entre les colonnes des deux DataFrames
df_diff = df_1.iloc[:, 1:].copy()
df_diff.iloc[:, 2:] = (np.array(df_1.iloc[:, 3:])-1) - (np.array(df_2.iloc[:, 3:])-1)
data_diff_1 = df_diff[df_diff["metric"] == "d2_absolute_log_error_score_by_sp"].iloc[:, 2:]
data_diff_2 = df_diff[df_diff["metric"]=="R2_Species_log"].iloc[:, 2:]

# Renommer les colonnes avec les stats
path_dir = "../inputs/data_RF_RSL/"
data = "Galaxy117-Sort_on_data_82_n"
df = pd.read_csv(path_dir + "/" + data + ".csv", sep=',')
df = df.iloc[:, 9:]
results = df.apply(stats)
sp_stats = results.loc['occurrences'].tolist()
sp_stats = (np.round(np.array(sp_stats)/406*100)).tolist()

concatenated_names = [f"{name1} ({name2}%)" for name1, name2 in zip(data_diff_1.columns, sp_stats)]
concatenated_names = [item.replace('.0%', '%') for item in concatenated_names]
data_diff_1.columns = concatenated_names
data_diff_2.columns = concatenated_names

# Réorganiser les colonnes par ordre décroissant des occurrences
ordre = sorted(range(len(sp_stats)), key=lambda k: sp_stats[k], reverse=True)
data_diff_1 = data_diff_1.iloc[:, ordre]
data_diff_2 = data_diff_2.iloc[:, ordre]

# Créer la figure avec deux sous-figures
plt.figure(figsize=(8, 8))

# Premier boxplot (original)
plt.subplot(1, 1, 1)
sns.boxplot(data=data_diff_1, orient='h', palette=['#1f77b4']* 47)
plt.xlabel('D2 Absolute Log Error (D2log) deviation\n by species between CNN-SDM with transfer learning and Random Forest')
plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
#plt.axvline(x=0.5, color='red', linestyle=':', linewidth=2)
#plt.axvline(x=-0.5, color='red', linestyle=':', linewidth=2)

#
plt.annotate('', xy=(0.05, 1.025), xycoords='axes fraction', xytext=(0.45, 1.025), 
            arrowprops=dict(arrowstyle="->", color='red', linewidth=2.5))
plt.annotate('', xy=(0.95, 1.025), xycoords='axes fraction', xytext=(0.55, 1.025), 
            arrowprops=dict(arrowstyle="->", color='red', linewidth=2.5))
plt.text(-2, -2.5 , "CNN-SDM more efficient", ha='center', fontsize=12, color='red')
plt.text(2, -2.5 , "RF more efficient", ha='center', fontsize=12, color='red')
plt.xlim(-4, 4)


'''
# Deuxième boxplot (par exemple, vous pouvez utiliser une autre colonne ou un autre DataFrame)
# Pour illustrer, je vais réutiliser les mêmes données différemment
plt.subplot(1, 2, 2)
sns.boxplot(data=data_diff_2, orient='h', palette=['#1f77b4'])
plt.xlabel('Difference between R-squared on Log-transformed data (R2log)\n by species between ResNet and RF')
plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
# Masquer les labels de l'axe y pour le deuxième boxplot
plt.gca().set_yticklabels([])
#
plt.annotate('', xy=(0.05, 1.025), xycoords='axes fraction', xytext=(0.45, 1.025), 
            arrowprops=dict(arrowstyle="->", color='red', linewidth=2.5))
plt.annotate('', xy=(0.95, 1.025), xycoords='axes fraction', xytext=(0.55, 1.025), 
            arrowprops=dict(arrowstyle="->", color='red', linewidth=2.5))
plt.text(-1, -2.5 , "ResNet more efficient", ha='center', fontsize=12, color='red')
plt.text(1, -2.5 , "RF more efficient", ha='center', fontsize=12, color='red')
plt.xlim(-2, 2)
'''


# Ajuster l'espacement entre les sous-figures
plt.tight_layout()
date = datetime.now().strftime("%Y_%m_%d")
plt.savefig(f'../outputs/abundances_results/figure_3_{date}.jpeg', dpi=300)

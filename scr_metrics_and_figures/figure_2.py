import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Charger les données
df = pd.read_csv("../outputs/abundances_results/metric.csv")

# Calculer les écarts types dynamiquement
def calculate_std(series_list, df):
    return {series: round(np.std(df[series]), 2) for series in series_list}

# Préparer les données pour les boxplots avec violons
def prepare_data(df, column_mapping):
    data_frames = {}
    for key, columns in column_mapping.items():
        df_temp = df[columns].copy()
        df_temp.columns = [key + f"_{col}" for col in df_temp.columns]
        data_frames[key] = df_temp
    return data_frames

# Graphiques
def plot_violin(ax, data, palette, y_label, x_sub_labels):
    sns.violinplot(ax=ax, data=data, density_norm="count", width=0.5, cut=0, linewidth=2, palette=palette, inner_kws=dict(box_width=15, whis_width=2), fill=False)
    ax.set_ylabel(y_label, labelpad=10, fontsize=12)  # Augmenter la taille de la légende Y
    ax.set_xticks(range(len(x_sub_labels)))
    ax.set_xticklabels(x_sub_labels, fontsize=12)  # Augmenter la taille des labels X

def annotate_std(ax, std_devs, palette, y_offset):
    for violin, (series, sd) in zip(ax.collections, std_devs.items()):
        x = violin.get_paths()[0].vertices[:, 0].mean()
        y = violin.get_paths()[0].vertices[:, 1].min() - y_offset
        ax.text(x, y, f"std = {sd}", ha='center', fontsize=12, color=palette[list(std_devs.keys()).index(series)])

def main():
    # Définir les colonnes et les labels
    prefixes_models = {
        '1': 'CNN_SDM_without_tl',
        '2': 'CNN_SDM_with_tl',
        '3': 'RF',
        }
    prefixes_metrics = {
        '1': 'd2_absolute_log_error_score',
        '2': 'spearman_coef',
        '3': 'R2_log_score',
        }

    # Définir les colonnes et les labels en utilisant les préfixes
    column_mapping = {
        'D2': [
            f'{prefixes_models["1"]}_{prefixes_metrics["1"]}',
            f'{prefixes_models["2"]}_{prefixes_metrics["1"]}',
            f'{prefixes_models["3"]}_{prefixes_metrics["1"]}',
            #f'{prefixes_models["4"]}_{prefixes_metrics["2"]}',
            #f'{prefixes_models["5"]}_{prefixes_metrics["2"]}'
            ],
        'Spearman': [
            f'{prefixes_models["1"]}_{prefixes_metrics["2"]}',
            f'{prefixes_models["2"]}_{prefixes_metrics["2"]}',
            f'{prefixes_models["3"]}_{prefixes_metrics["2"]}',
            #f'{prefixes_models["4"]}_{prefixes_metrics["3"]}',
            #f'{prefixes_models["5"]}_{prefixes_metrics["3"]}'
            ],
        'R2': [
            f'{prefixes_models["1"]}_{prefixes_metrics["3"]}',
            f'{prefixes_models["2"]}_{prefixes_metrics["3"]}',
            f'{prefixes_models["3"]}_{prefixes_metrics["3"]}',
            #f'{prefixes_models["4"]}_{prefixes_metrics["1"]}',
            #f'{prefixes_models["5"]}_{prefixes_metrics["1"]}'
            ],
        }
    

    # Préparer les données
    data_frames = prepare_data(df, column_mapping)

    # Calculer les écarts types
    std_devs = {
        'D2': calculate_std(column_mapping['D2'], df),
        'Spearman': calculate_std(column_mapping['Spearman'], df),
        'R2': calculate_std(column_mapping['R2'], df),
    }

    # Création des graphiques
    fig, axes = plt.subplots(1, 3, figsize=(17, 8))

    # Liste des couleurs
    colors = ["black", "black", "black"]

    # Tracer les graphiques
    models = [
        'CNN-SDM\nwithout\ntransfer\nlearning',
        'CNN-SDM\nwith\ntransfer\nlearning',
        'Random\nForest',
        ]

    plot_violin(axes[0], data_frames['D2'], colors,
                'D-squared regression score fonction on log-transformed data (D2Log)',
                models)
    
    plot_violin(axes[1], data_frames['Spearman'], colors,
                "Spearman's rank correlation coefficient (Spearman coeff)",
                models)
    
    plot_violin(axes[2], data_frames['R2'], colors, 
                'R-squared regression score fonction on log-transformed data (R2Log)',
                models)
    # Ajouter les annotations pour les écarts types
    annotate_std(axes[0], std_devs['D2'], colors, 0.0075)
    annotate_std(axes[1], std_devs['Spearman'], colors, 0.0075)
    annotate_std(axes[2], std_devs['R2'], colors, 0.015)

    # Ajouter les labels des sous-graphiques
    for i, ax in enumerate(axes):
        ax.text(-0.15, 1, chr(65 + i), transform=ax.transAxes, fontsize=23, va='top')

    plt.subplots_adjust(wspace=0.5)  # Ajuster l'espace entre les sous-graphiques
    plt.tight_layout()

    # Sauvegarder l'image
    date = datetime.now().strftime("%Y_%m_%d")
    plt.savefig(f'../outputs/abundances_results/figure_2_{date}.jpeg', dpi=300)

if __name__ == "__main__":
    main()
#%% Imports
import os
import pandas as pd
import numpy as np
import shapely.geometry
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as cx
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import Normalize
from tqdm import tqdm
from matplotlib.colors import TwoSlopeNorm
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import d2_absolute_error_score
from datetime import datetime

#%% Function
def df_by_fold(fold, path_RF_pred_dir, path_CNN_SDM_with_tl_pred_dir):
    """
    Load and merge two DataFrames on specified columns.
    """
    # Load the CSV file into a DataFrame
    df_RF_pred = pd.read_csv(f'{path_RF_pred_dir}/pred_flod_{fold}_stacked.csv')
    df_CNN_SDM_with_tl_pred = pd.read_csv(f'{path_CNN_SDM_with_tl_pred_dir}/pred_flod_{fold}_stacked.csv')
    # Renaming the 'pred' column to avoid conflicts
    df_RF_pred.rename(columns={'pred': 'pred_RF'}, inplace=True)
    df_CNN_SDM_with_tl_pred.rename(columns={'pred': 'pred_CNN_tl'}, inplace=True)
    # Merge the DataFrames on the specified columns on SurveyID and specie
    df_marge = pd.merge(df_RF_pred, df_CNN_SDM_with_tl_pred[['SurveyID', 'specie', 'pred_CNN_tl']], on=['SurveyID', 'specie'])
    # Reorder columns to place 'pred_CNN_tl' after 'pred_RF'
    cols = df_marge.columns.tolist()
    cols.insert(cols.index('pred_RF') + 1, cols.pop(cols.index('pred_CNN_tl')))
    df_marge = df_marge[cols]
    # Add fold column after the 'specie' column
    df_marge.insert(df_marge.columns.get_loc('specie') + 1, 'fold', fold)
    return df_marge

def Robinson_grid_creation(df, cell_size_km):
    xmin = min(df.geometry.x)
    xmax = max(df.geometry.x)
    ymin = min(df.geometry.y)
    ymax = max(df.geometry.y)
    cell_size = cell_size_km * 1000
    xs = np.arange(xmin, xmax, cell_size)
    ys = np.arange(ymin, ymax, cell_size)
    xx, yy = np.meshgrid(xs, ys, sparse=False)
    # create the cells in a loop
    grid_cells = []
    for x0 in np.arange(xmin, xmax+cell_size, cell_size):
        for y0 in np.arange(ymin, ymax+cell_size, cell_size):
            # bounds
            x1 = x0-cell_size
            y1 = y0+cell_size
            grid_cells.append(shapely.geometry.box(x0, y0, x1, y1))
    # convert in GeoDataFrame
    cells = gpd.GeoDataFrame(grid_cells,
                            columns = ['geometry'],
                            crs = '+proj=robin')
    # Renames index
    cells.index.set_names('cell_index', inplace=True)
    xmin, ymin, xmax, ymax = cells.total_bounds
    return cells, xmin, ymin, xmax, ymax


def compute_df_dif(df_pred, df_true):
    df_dif = df_true.copy()
    for col in df_true.columns:
        if col != 'cell_index' and col != 'color':
            df_dif[col] = df_pred[col] - df_true[col]
    return df_dif


def TicksAsymetriques(n_ticks_total, global_min_dif, global_max_dif):
    """
    Génère des ticks asymétriques pour une colorbar.
    
    :param n_ticks_total: Nombre total de ticks à générer.
    :param global_min_dif: Valeur minimale de la différence globale.
    :param global_max_dif: Valeur maximale de la différence globale.
    :return: Ticks asymétriques pour la colorbar.
    """
    # Proportion de ticks négatifs vs positifs
    neg_ratio = abs(global_min_dif) / (abs(global_min_dif) + abs(global_max_dif))
    n_ticks_neg = max(3, int(n_ticks_total * neg_ratio))  # au moins 2 pour avoir vmin et 0
    n_ticks_pos = max(3, n_ticks_total - n_ticks_neg)

    # Génération des ticks asymétriques
    neg_ticks = np.linspace(global_min_dif, 0, n_ticks_neg, endpoint=False)
    pos_ticks = np.linspace(0, global_max_dif, n_ticks_pos)
    
    return np.concatenate([neg_ticks, pos_ticks])


def Figure_pred(df_true,
                df_pred_1,
                df_pred_2,
                df_dif_1,
                df_dif_2,
                sp_percent_occ,
                sp_percent_occ_local,
                retained_cells,
                sp,
                xmin, xmax, ymin, ymax,
                output_path_fig_2):

    min_df_true = df_true[sp].min()
    max_df_true = df_true[sp].max()
    min_df_pred_1 = df_pred_1[sp].min()
    max_df_pred_1 = df_pred_1[sp].max()
    min_df_pred_2 = df_pred_2[sp].min()
    max_df_pred_2 = df_pred_2[sp].max()
    global_min = min(min_df_true, min_df_pred_1, min_df_pred_2)
    global_max = max(max_df_true, max_df_pred_1, max_df_pred_2)
    if global_max == 0:
        global_max = 0.01  # Avoid division by zero in normalization
    norm = Normalize(vmin=global_min, vmax=global_max)
    min_df_dif_1 = df_dif_1[sp].min()
    max_df_dif_1 = df_dif_1[sp].max()
    min_df_dif_2 = df_dif_2[sp].min()
    max_df_dif_2 = df_dif_2[sp].max()
    global_min_dif = min(min_df_dif_1, min_df_dif_2)
    global_max_dif = max(max_df_dif_1, max_df_dif_2)
    
    
    

    global_min_dif = -global_max_dif
    
    
    
    
    
    if global_min_dif > 0:
        global_min_dif = 0
    if global_max_dif < 0:
        global_max_dif = 0
    if global_max_dif == 0 and global_min_dif == 0:
        global_max_dif = 0.01  # Avoid division by zero in normalization
        global_min_dif = -0.01  # Avoid division by zero in normalization
    if global_min_dif < 0 and global_max_dif > 0:
        # Use TwoSlopeNorm to center the colormap at zero
        norm_dif = TwoSlopeNorm(vmin=global_min_dif, vcenter=0, vmax=global_max_dif)
    else :
        norm_dif = Normalize(vmin=global_min_dif, vmax=global_max_dif)
    
    ### Visualization of species presence/absence in a 3x2 grid layout
    fig, axes = plt.subplots(3, 2, figsize=(14, 18))#, sharex=True, sharey=True)
    axes[0, 0].remove()  # Remove the first cell in the first row
    fig.subplots_adjust(left=0, right=0.85, top=1, bottom=0, wspace=0.1, hspace=0.15)
    # Set the figure title
    # Format species name in italic, preserving spaces
    sp_italic = r"$\it{" + sp.replace(" ", r"\ ") + "}$"
    fig.text(0.20, 0.75,
             f"{sp_italic}\n\n-\n\nMean of log abundance by cell in\n Robinson Coordinate Reference System\n(grid of {cell_size_km} x {cell_size_km} km)\n\n-\n\nOccurence percentage for\nthe entire abondance dataset: {sp_percent_occ:.1f}%\n\nOccurence percentage for\nthe abondance dataset in this area: {sp_percent_occ_local:.1f}%",
             ha='center', fontsize=16)
    # Plot 1 (centered on the first row)
    ax = axes[0, 1]
    ax.set_title("True values", fontsize=13)
    df_true['color'] = df_true[sp].apply(lambda x: plt.cm.RdYlGn(norm(x)))
    cell_colors = df_true.set_index('cell_index')['color']
    retained_cells.plot(ax=ax, facecolor=retained_cells.index.map(cell_colors), edgecolor='black')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel("Y Coordinates (in $10^6$ km)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f'{x*x_a:.{x_b}f}'))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, pos: f'{y*y_a:.{y_b}f}'))
    # Add a white rectangle behind the color bar
    ax.add_patch(plt.Rectangle((0, 1), 0.24, -0.4,
                                transform=ax.transAxes,
                                color='white', alpha=0.7,
                                zorder=2))
    # Create an inset axis for the color bar
    cax = inset_axes(ax, width="2%", height="30%", loc='upper left', borderpad=2)
    sm = plt.cm.ScalarMappable(cmap='RdYlGn', norm=plt.Normalize(vmin=global_min, vmax=global_max))
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_label('mean abundance')
    # Add world map in background
    cx.add_basemap(ax, crs=cells.crs,source=cx.providers.OpenStreetMap.Mapnik)

    # Plot 2 (second row, left)
    ax = axes[1, 0]
    ax.set_title("Random Forest:\nPredicted values", fontsize=13)
    df_pred_1['color'] = df_pred_1[sp].apply(lambda x: plt.cm.RdYlGn(norm(x)))
    cell_colors = df_pred_1.set_index('cell_index')['color']
    retained_cells.plot(ax=ax, facecolor=retained_cells.index.map(cell_colors), edgecolor='black')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel("Y Coordinates (in $10^6$ km)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f'{x*x_a:.{x_b}f}'))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, pos: f'{y*y_a:.{y_b}f}'))
    # Add a white rectangle behind the color bar
    ax.add_patch(plt.Rectangle((0, 1), 0.24, -0.4,
                                transform=ax.transAxes,
                                color='white', alpha=0.7,
                                zorder=2))
    # Create an inset axis for the color bar
    cax = inset_axes(ax, width="2%", height="30%", loc='upper left', borderpad=2)
    sm = plt.cm.ScalarMappable(cmap='RdYlGn', norm=plt.Normalize(vmin=global_min, vmax=global_max))
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_label('mean abundance')
    # Add world map in background
    cx.add_basemap(ax, crs=cells.crs,source=cx.providers.OpenStreetMap.Mapnik)

    # Plot 3 (third row, left)
    ax = axes[2, 0]
    ax.set_title("Random Forest:\nDifference between true and predicted values", fontsize=13)
    if global_min_dif < 0 and global_max_dif > 0:
        df_dif_1['color'] = df_dif_1[sp].apply(lambda x: plt.cm.bwr((norm_dif(x))))
    elif global_min_dif == 0:
        bwr = plt.get_cmap('bwr')
        white_red = LinearSegmentedColormap.from_list('white_red', bwr(np.linspace(0.5, 1.0, 256)))
        df_dif_1['color'] = df_dif_1[sp].apply(lambda x: white_red((norm_dif(x))))
    elif global_max_dif == 0:
        bwr = plt.get_cmap('bwr')
        blue_white = LinearSegmentedColormap.from_list('blue_white', bwr(np.linspace(0, 0.5, 256)))
        df_dif_1['color'] = df_dif_1[sp].apply(lambda x: blue_white((norm_dif(x))))
    cell_colors = df_dif_1.set_index('cell_index')['color']
    retained_cells.plot(ax=ax, facecolor=retained_cells.index.map(cell_colors), edgecolor='black')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f'{x*x_a:.{x_b}f}'))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, pos: f'{y*y_a:.{y_b}f}'))
    ax.set_xlabel("X Coordinates (in $10^6$ km)")
    ax.set_ylabel("Y Coordinates (in $10^6$ km)")
    # Add a white rectangle behind the color bar
    ax.add_patch(plt.Rectangle((0, 1), 0.24, -0.4,
                                transform=ax.transAxes,
                                color='white', alpha=0.7,
                                zorder=2))
    # Create an inset axis for the color bar
    cax = inset_axes(ax, width="2%", height="30%", loc='upper left', borderpad=2)
    # Set the color normalization so that 0 is mapped to white in the 'bwr' colormap, even if vmin and vmax are not symmetric
    if global_min_dif < 0 and global_max_dif > 0:
        sm = plt.cm.ScalarMappable(cmap='bwr', norm=norm_dif)
        sm.set_array([])
        ticks = TicksAsymetriques(7, global_min_dif, global_max_dif)
        cbar = plt.colorbar(sm, cax=cax,ticks=ticks)
        cbar.ax.set_yticklabels([f"{t:.2f}" for t in ticks])  # format facultatif
    elif global_min_dif == 0:
        sm = plt.cm.ScalarMappable(cmap=white_red, norm=norm_dif)
        sm.set_array([])
        cbar = plt.colorbar(sm, cax=cax)
    elif global_max_dif == 0:
        sm = plt.cm.ScalarMappable(cmap=blue_white, norm=norm_dif)       
        sm.set_array([])
        cbar = plt.colorbar(sm, cax=cax)
    cbar.set_label('diff. of mean abundance')
    # Add world map in background
    cx.add_basemap(ax, crs=cells.crs,source=cx.providers.OpenStreetMap.Mapnik)

    # Plot 4 (second row, right)
    ax = axes[1, 1]
    ax.set_title("CNN-SDM with transfer learning:\nPredicted values", fontsize=13)
    df_pred_2['color'] = df_pred_2[sp].apply(lambda x: plt.cm.RdYlGn(norm(x)))
    cell_colors = df_pred_2.set_index('cell_index')['color']
    retained_cells.plot(ax=ax, facecolor=retained_cells.index.map(cell_colors), edgecolor='black')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f'{x*x_a:.{x_b}f}'))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, pos: f'{y*y_a:.{y_b}f}'))
    # Add a white rectangle behind the color bar
    ax.add_patch(plt.Rectangle((0, 1), 0.24, -0.4,
                                transform=ax.transAxes,
                                color='white', alpha=0.7,
                                zorder=2))
    # Create an inset axis for the color bar
    cax = inset_axes(ax, width="2%", height="30%", loc='upper left', borderpad=2)
    sm = plt.cm.ScalarMappable(cmap='RdYlGn', norm=plt.Normalize(vmin=global_min, vmax=global_max))
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_label('mean abundance')
    # Add world map in background
    cx.add_basemap(ax, crs=cells.crs,source=cx.providers.OpenStreetMap.Mapnik)

    # Plot 5 (third row, right)
    ax = axes[2, 1]
    ax.set_title("CNN-SDM with transfer learning:\nDifference between true and predicted values", fontsize=13)
    if global_min_dif < 0 and global_max_dif > 0:
        df_dif_2['color'] = df_dif_2[sp].apply(lambda x: plt.cm.bwr((norm_dif(x))))
    elif global_min_dif == 0:
        bwr = plt.get_cmap('bwr')
        white_red = LinearSegmentedColormap.from_list('white_red', bwr(np.linspace(0.5, 1.0, 256)))
        df_dif_2['color'] = df_dif_2[sp].apply(lambda x: white_red((norm_dif(x))))
    elif global_max_dif == 0:
        bwr = plt.get_cmap('bwr')
        blue_white = LinearSegmentedColormap.from_list('blue_white', bwr(np.linspace(0, 0.5, 256)))
        df_dif_2['color'] = df_dif_2[sp].apply(lambda x: blue_white((norm_dif(x))))
    cell_colors = df_dif_2.set_index('cell_index')['color']
    retained_cells.plot(ax=ax, facecolor=retained_cells.index.map(cell_colors), edgecolor='black')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f'{x*x_a:.{x_b}f}'))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, pos: f'{y*y_a:.{y_b}f}'))
    ax.set_xlabel("X Coordinates (in $10^6$ km)")
    # Add a white rectangle behind the color bar
    ax.add_patch(plt.Rectangle((0, 1), 0.24, -0.4,
                                transform=ax.transAxes,
                                color='white', alpha=0.7,
                                zorder=2))
    # Create an inset axis for the color bar
    cax = inset_axes(ax, width="2%", height="30%", loc='upper left', borderpad=2)
    # Set the color normalization so that 0 is mapped to white in the 'bwr' colormap, even if vmin and vmax are not symmetric
    if global_min_dif < 0 and global_max_dif > 0:
        sm = plt.cm.ScalarMappable(cmap='bwr', norm=norm_dif)
        sm.set_array([])
        ticks = TicksAsymetriques(7, global_min_dif, global_max_dif)
        cbar = plt.colorbar(sm, cax=cax,ticks=ticks)
        cbar.ax.set_yticklabels([f"{t:.2f}" for t in ticks])  # format facultatif
    elif global_min_dif == 0:
        sm = plt.cm.ScalarMappable(cmap=white_red, norm=norm_dif)
        sm.set_array([])
        cbar = plt.colorbar(sm, cax=cax)        
    elif global_max_dif == 0:
        sm = plt.cm.ScalarMappable(cmap=blue_white, norm=norm_dif)    
        sm.set_array([])
        cbar = plt.colorbar(sm, cax=cax)
    cbar.set_label('diff. of mean abundance')
    # Add world map in background
    cx.add_basemap(ax, crs=cells.crs,source=cx.providers.OpenStreetMap.Mapnik)

    # Save the figure
    plt.savefig(output_path_fig_2, bbox_inches='tight', dpi=300)
    plt.close(fig)  # Fermer la figure pour éviter les fuites de mémoire et les warnings


#%% Settings
# Directories paths
path_output_dir_0 = '../outputs/abundances_results/pred_maps/'
path_RF_pred_dir ='../outputs/pred_RF_abundance/'
path_CNN_SDM_with_tl_pred_dir = '../outputs/pred_cnn_sdm_abundance_with_tl/'
# Files paths
path_meta_data_file = '../inputs/data_RF_RSL/Galaxy117-Sort_on_data_82_n.csv'
path_metric_by_sp_file = '../outputs/abundances_results/metric_by_sp.csv'
# Values
fold_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]  # List of folds to process
# predefined map type 
map_type = 'Mallorca_Ibiza_Zoom_4_4' #Alboran_Sea_5_5 ; #West_Med_15_15 ; Mallorca_Ibiza_5_5 ; Perpignan_Mataro_2_2 ; Alboran_Sea_Zoom_6_6 ; Mallorca_Ibiza_Zoom_4_4
# Test map
test_map = True # True if you want to test the map with a single species, False to plot all species
test_name = 'Serranus cabrilla' #'Apogon imberbis'#'Serranus scriba'

if map_type == 'West_Med_15_15':
    path_output_dir = f'{path_output_dir_0}{map_type}/'
    cell_size_km=15 # Size of the grid cells in kilometers
    # Define the limits for the x and y coordinates (no limits if set to 'no')
    x_min_limit = 'no'
    x_max_limit = 0.48* 10**6 #'no'
    y_min_limit = 3.70 * 10**6 #'no'
    y_max_limit = 'no'
    # Set x-axis in scientific notation
    x_a = 1e-6 # set x-axis in millions
    x_b = '1' # nomber of decimals
    # Set y-axis in scientific notation
    y_a = 1e-6 # set y-axis in millions
    y_b = '1' # nomber of decimals
elif map_type == 'Mallorca_Ibiza_5_5':
    path_output_dir = f'{path_output_dir_0}{map_type}/'
    cell_size_km=5 # Size of the grid cells in kilometers
    # Define the limits for the x and y coordinates (no limits if set to 'no')
    x_min_limit = 0.09 * 10**6 #'no'
    x_max_limit = 0.31 * 10**6 #'no'
    y_min_limit = 4.1 * 10**6 #'no'
    y_max_limit = 4.3 * 10**6 #'no'
    # Set x-axis in scientific notation
    x_a = 1e-6 # set x-axis in millions
    x_b = '3' # nomber of decimals
    # Set y-axis in scientific notation
    y_a = 1e-6 # set y-axis in millions
    y_b = '3' # nomber of decimals
elif map_type == 'X':
    path_output_dir = f'{path_output_dir_0}{map_type}/'
    cell_size_km=4 # Size of the grid cells in kilometers
    # Define the limits for the x and y coordinates (no limits if set to 'no')
    x_min_limit = -0.1 * 10**6 #'no'
    x_max_limit = 0.1 * 10**6 #'no'
    y_min_limit = 4.0 * 10**6 #'no'
    y_max_limit = 4.2 * 10**6 #'no'
    # Set x-axis in scientific notation
    x_a = 1e-6 # set x-axis in millions
    x_b = '3' # nomber of decimals
    # Set y-axis in scientific notation
    y_a = 1e-6 # set y-axis in millions
    y_b = '3' # nomber of decimals
elif map_type == 'Ibiza_5_5':
    path_output_dir = f'{path_output_dir_0}{map_type}/'
    cell_size_km=2 # Size of the grid cells in kilometers
    # Define the limits for the x and y coordinates (no limits if set to 'no')
    x_min_limit = 0.2 * 10**6 #'no'
    x_max_limit = 0.31 * 10**6 #'no'
    y_min_limit = 4.18 * 10**6 #'no'
    y_max_limit = 4.29 * 10**6 #'no'
    # Set x-axis in scientific notation
    x_a = 1e-6 # set x-axis in millions
    x_b = '3' # nomber of decimals
    # Set y-axis in scientific notation
    y_a = 1e-6 # set y-axis in millions
    y_b = '3' # nomber of decimals
elif map_type == 'Mallorca_Ibiza_Zoom_4_4':
    path_output_dir = f'{path_output_dir_0}{map_type}/'
    cell_size_km=5 # Size of the grid cells in kilometers
    # Define the limits for the x and y coordinates (no limits if set to 'no')
    x_min_limit = 0.095 * 10**6 #0.075 * 10**6 #'no'
    x_max_limit = 0.265 * 10**6 #0.150 * 10**6 #'no'
    y_min_limit = 4.125 * 10**6 #4.125 * 10**6 #'no'
    y_max_limit = 4.285 * 10**6 #4.200 * 10**6 #'no'
    # Set x-axis in scientific notation
    x_a = 1e-6 # set x-axis in millions
    x_b = '2' # nomber of decimals
    # Set y-axis in scientific notation
    y_a = 1e-6 # set y-axis in millions
    y_b = '2' # nomber of decimals
elif map_type == 'Alboran_Sea_Zoom_6_6':
    path_output_dir = f'{path_output_dir_0}{map_type}/'
    cell_size_km=6 # Size of the grid cells in kilometers
    # Define the limits for the x and y coordinates (no limits if set to 'no')
    x_min_limit = -0.50 * 10**6 #'no'
    x_max_limit = -0.225 * 10**6 #'no'
    y_min_limit = 3.75 * 10**6 #'no'
    y_max_limit = 4.025 * 10**6 #'no'
    # Set x-axis in scientific notation
    x_a = 1e-6 # set x-axis in millions
    x_b = '3' # nomber of decimals
    # Set y-axis in scientific notation
    y_a = 1e-6 # set y-axis in millions
    y_b = '3' # nomber of decimals
elif map_type == 'Alboran_Sea_5_5':
    path_output_dir = f'{path_output_dir_0}{map_type}/'
    cell_size_km=5 # Size of the grid cells in kilometers
    # Define the limits for the x and y coordinates (no limits if set to 'no')
    x_min_limit = -0.50 * 10**6 #'no'
    x_max_limit = -0.05 * 10**6 #'no'
    y_min_limit = 3.70 * 10**6 #'no'
    y_max_limit = 4.15 * 10**6 #'no'
    # Set x-axis in scientific notation
    x_a = 1e-6 # set x-axis in millions
    x_b = '3' # nomber of decimals
    # Set y-axis in scientific notation
    y_a = 1e-6 # set y-axis in millions
    y_b = '3' # nomber of decimals
elif map_type == 'Perpignan_Mataro_2_2':
    path_output_dir = f'{path_output_dir_0}{map_type}/'
    cell_size_km=2 # Size of the grid cells in kilometers
    # Define the limits for the x and y coordinates (no limits if set to 'no')
    x_min_limit = 0.2* 10**6 #'no'
    x_max_limit = 0.3* 10**6 #'no'
    y_min_limit = 4.45* 10**6 #'no'
    y_max_limit = 4.55* 10**6 #'no'
    # Helper to set x-axis in millions
    # Set x-axis in scientific notation
    x_a = 1e-6 # set x-axis in millions
    x_b = '3' # nomber of decimals
    # Set y-axis in scientific notation
    y_a = 1e-6 # set y-axis in millions
    y_b = '3' # nomber of decimals


#%% Main script
## Create output directory if it does not exist
if not os.path.exists(path_output_dir_0):
    os.makedirs(path_output_dir_0)
## Load data
df_meta_data = pd.read_csv(path_meta_data_file)
df_metric_by_sp = pd.read_csv(path_metric_by_sp_file)
df_list = []
for fold in fold_list:
    df_fold = df_by_fold(fold, path_RF_pred_dir, path_CNN_SDM_with_tl_pred_dir)
    df_list.append(df_fold)
df = pd.concat(df_list, axis=0).reset_index(drop=True)

# Convertir 'true' en exp(y) - 1
df['true'] = np.round(np.exp(df['true']) - 1)
# Convertir 'pred_RF' et 'pred_CNN_tl' en exp(y) - 1
df['pred_RF'] = np.exp(df['pred_RF']) - 1
df['pred_RF'] = np.where(df['pred_RF'] - np.floor(df['pred_RF']) >= 0.05, np.ceil(df['pred_RF']), np.floor(df['pred_RF']))
df['pred_CNN_tl'] = np.exp(df['pred_CNN_tl']) - 1
df['pred_CNN_tl'] = np.where(df['pred_CNN_tl'] - np.floor(df['pred_CNN_tl']) >= 0.05, np.ceil(df['pred_CNN_tl']), np.floor(df['pred_CNN_tl']))

'''
# Convertir 'true' en exp(y) - 1
df['true'] = np.round(np.exp(df['true']) - 1)
# Convertir 'pred_RF' et 'pred_CNN_tl' en exp(y) - 1
df['pred_RF'] = np.round(np.exp(df['pred_RF']) - 1)
df['pred_CNN_tl'] = np.round(np.exp(df['pred_CNN_tl']) - 1)
# reconvertir en log+1
df['true'] = np.log1p(df['true'])
df['pred_RF'] = np.log1p(df['pred_RF'])
df['pred_CNN_tl'] = np.log1p(df['pred_CNN_tl'])
'''
# Add 'SurveyDate' column
for id in df['SurveyID'].unique():
    df.loc[df['SurveyID'] == id, 'SurveyDate'] = df_meta_data[df_meta_data['SurveyID'] == id]['SurveyDate'].values[0]
# Convertir 'SurveyDate' en format datetime
df['SurveyDate'] = pd.to_datetime(df['SurveyDate']).dt.strftime('%Y-%m-%d')


df.insert(df.columns.tolist().index('SurveyID') + 1, 'geometry', np.nan)
## Recover the geometry from df_meta_data
for id in df['SurveyID'].unique():
    df.loc[df['SurveyID'] == id, 'geometry'] = df_meta_data[df_meta_data['SurveyID'] == id]['geom'].values[0]
## Convert the 'geometry' column from WKT to GeoSeries
df['geometry'] = gpd.GeoSeries.from_wkt(df['geometry'])
## Convert to the Robinson projection
geometry = gpd.GeoSeries(df['geometry'], crs='EPSG:4326')
geometry_rob = geometry.to_crs('+proj=robin')
df = gpd.GeoDataFrame(df, geometry=geometry_rob)
'''
df = (df
      .groupby(['geometry','specie','subset','SurveyDate'], as_index=False)
      .agg({'SurveyID': 'first',
            'true': 'min',
            'pred_RF': 'mean',
            'pred_CNN_tl': 'mean',}))
'''
df = (df
      .groupby(['SurveyID','specie','subset',], as_index=False)
      .agg({'geometry': 'first',
            'true': 'first',
            'pred_RF': 'mean',
            'pred_CNN_tl': 'mean',}))

df = gpd.GeoDataFrame(df, geometry=df['geometry'])

## Filter the DataFrame to keep only the 'test' subset
df_test = df[df['subset'] == 'test']

## Apply the limits to the DataFrame
if x_min_limit != 'no':
    df_test = df_test[df_test.geometry.x > x_min_limit].copy()
if x_max_limit != 'no':
    df_test = df_test[df_test.geometry.x < x_max_limit].copy()
if y_min_limit != 'no':
    df_test = df_test[df_test.geometry.y > y_min_limit].copy()
if y_max_limit != 'no':
    df_test = df_test[df_test.geometry.y < y_max_limit].copy()

## Reset the index of df_test
df_test.reset_index(inplace=True)

## Robinson's grid creation
cells, xmin, ymin, xmax, ymax = Robinson_grid_creation(df_test, cell_size_km)

## Apply the limits to the DataFrame
if x_min_limit != 'no':
    xmin = x_min_limit
if x_max_limit != 'no':
    xmax = x_max_limit
if y_min_limit != 'no':
    ymin = y_min_limit
if y_max_limit != 'no':
    ymax = y_max_limit

## Filter the DataFrame to keep only the occurrences within the limits
df_local = df.copy()
df_local = df_local[df_local.geometry.x > xmin].copy()
df_local = df_local[df_local.geometry.x < xmax].copy()
df_local = df_local[df_local.geometry.y > ymin].copy()
df_local = df_local[df_local.geometry.y < ymax].copy()

## Define the cell index for each occurrence
# Spatial join of two GeoDataFrames
df_test = gpd.sjoin(df_test, cells, predicate="intersects", how='left')
# Get rid of duplicated occurrences intersecting two cells
df_test = df_test[~df_test.index.duplicated(keep='first')]
## Calculate the mean abundance for each cell for each species
# for pred Random Forest
mean_abundance_RF = df_test.pivot_table(
    index='cell_index', 
    columns='specie', 
    values='pred_RF', 
    aggfunc='mean'
).reset_index()
# for pred of CNN with transfer learning
mean_abundance_CNN_tl = df_test.pivot_table(
    index='cell_index', 
    columns='specie', 
    values='pred_CNN_tl', 
    aggfunc='mean'
).reset_index()
# for true values
mean_abundance_true = df_test.pivot_table(
    index='cell_index', 
    columns='specie', 
    values='true', 
    aggfunc='mean'
).reset_index()

# Calcul du d2_absolute_error_score pour chaque espèce


df_local_test = df_local[df_local['subset'] == 'test']

d2_scores = []
for sp in df_local_test['specie'].unique():
    y_true = df_local_test[df_local_test['specie'] == sp]['true']
    y_pred_CNN_tl = df_local_test[df_local_test['specie'] == sp]['pred_CNN_tl']
    y_pred_RF = df_local_test[df_local_test['specie'] == sp]['pred_RF']
    # Vérifier qu'il y a au moins deux valeurs pour calculer le score
    if len(y_true) > 1:
        d2_CNN_tl = d2_absolute_error_score(y_true, y_pred_CNN_tl)
        d2_RF = d2_absolute_error_score(y_true, y_pred_RF)
        d2_diff = (d2_RF-1)- (d2_CNN_tl-1)
    else:
        d2 = None
    d2_scores.append({'specie': sp,
                      'd2_CNN_tl': d2_CNN_tl,
                      'd2_RF': d2_RF,
                      'd2_diff': d2_diff})

# Convertir la liste en DataFrame
d2_scores_df = pd.DataFrame(d2_scores)
print(d2_scores_df)




# for differences between true and predicted values 
mean_abundance_CNN_tl_dif = compute_df_dif(mean_abundance_CNN_tl, mean_abundance_true)
mean_abundance_RF_dif = compute_df_dif(mean_abundance_RF, mean_abundance_true)
## Select cells with values
sel_cells_indexes = set(mean_abundance_true.cell_index)
retained_cells = cells[cells.index.isin(sel_cells_indexes)]
## Select the species to plot
species_list = df_test['specie'].unique().tolist()
if test_map == True:
    species_list = [test_name] #'Serranus cabrilla'
## Plotting
for sp in tqdm(species_list, desc="Processing species"):
    df_sp = df[df['specie'] == sp]['true']
    percentage_non_zero = (df_sp != 0).sum() / len(df_sp) * 100
    percentage_non_zero_rounded = round(percentage_non_zero)

    df_local_sp = df_local[df_local['specie'] == sp]['true']
    percentage_non_zero_local = (df_local_sp != 0).sum() / len(df_local_sp) * 100
    
    date = datetime.now().strftime("%Y_%m_%d")
    output_path_fig_2 = f'../outputs/abundances_results/figure_4_{date}.jpeg'

    Figure_pred(df_true=mean_abundance_true,
                df_pred_1=mean_abundance_RF,
                df_pred_2=mean_abundance_CNN_tl,
                df_dif_1=mean_abundance_RF_dif,
                df_dif_2=mean_abundance_CNN_tl_dif,
                sp_percent_occ=percentage_non_zero,
                sp_percent_occ_local=percentage_non_zero_local,
                retained_cells=retained_cells,
                sp=sp,
                xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
                output_path_fig_2=output_path_fig_2)


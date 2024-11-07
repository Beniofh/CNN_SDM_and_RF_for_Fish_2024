import pandas as pd
import numpy as np


# Chemin 
path_dir = '/home/bbourel/Documents/Fish-Predict/Gestion_datasets/datasets/Reef_Life_Survey/'
data = "Galaxy117-Sort_on_data_82_n_vec_subset"

path_dir_2 = "/home/bbourel/Documents/IA/malpolon/examples_perso/multiband_unimodel/outputs/cnn_multi_band_sortie_tf/tranfer_learning_rf/"
data_2 = "2023-10-06_15-09-58"

# récupération des données
# pour le type de vecteur
df = pd.read_csv(path_dir + "/" + data + ".csv", sep=',', index_col='SurveyID')
# pour les sorties du CNN
df_pred = pd.read_csv(path_dir_2 + "/" + data_2 + ".csv", sep=',', index_col='SurveyID')



SurveyID_val = df_pred.index[df_pred.subset=='val'] 
sp_liste = eval(df.columns[-5])
df_final = pd.DataFrame(columns=['taxon', 'longitude', 'latitude', 'date', 'y_pred', 'y'])

for SurveyID in SurveyID_val :
    for sp in sp_liste :
        df_temp = pd.DataFrame([[sp, df.SiteLong[SurveyID], df.SiteLat[SurveyID], df.SurveyDate[SurveyID] , df_pred[str(sp_liste.index(sp))][SurveyID], eval(df[df.columns[-5]][SurveyID])[sp_liste.index(sp)]]], 
                                columns=['taxon', 'longitude', 'latitude', 'date', 'y_pred', 'y'])
        df_final= df_final.append(df_temp, ignore_index=True)
        

df_final = df_final.astype({'taxon': str,
                            'longitude': float,
                            'latitude': float,
                            'date' : str,
                            'y_pred': float,
                            'y': float,
                            })

df_final['y_pred_log(n+1)'] = df_final.y_pred

df_final.y_pred = np.exp(df_final.y_pred)-1

df_final['y_log(n+1)'] = np.log((df_final.y)+1)

# a faire eb fonction des besions
#df_final.abondance_true = round((np.exp(df_final.abondance_true)-1),0).astype(int)



df_final.to_csv(path_dir_2 + "/" + data_2 + "_compil.csv", sep='\t', index=False)

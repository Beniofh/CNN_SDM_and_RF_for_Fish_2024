import zipfile
import os

def unzip_and_move(zip_file_path, target_dir):
    # Décompression du fichier .zip
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(target_dir)
    # Suppression du dossier temporaire
    # shutil.rmtree(temp_dir)

def unzip_and_move(zip_files, target_dir):
    # Création du dossier cible s'il n'existe pas
    os.makedirs(target_dir, exist_ok=True)
    for zip_file_path in zip_files:
        # Décompression du fichier .zip
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
        # Suppression du fichier .zip
        #os.remove(zip_file_path)


zip_file = ['inputs/outputs_cnn_sdm_abundance_without_tl_1.zip',
            'inputs/outputs_cnn_sdm_abundance_without_tl_2.zip'] 
destination_folder = 'inputs/outputs_cnn_sdm_abundance_without_tl'  
unzip_and_move(zip_file, destination_folder)


zip_file = ['inputs/outputs_cnn_sdm_abundance_with_tl_1.zip',
            'inputs/outputs_cnn_sdm_abundance_with_tl_2.zip']
destination_folder = 'inputs/outputs_cnn_sdm_abundance_with_tl'  
unzip_and_move(zip_file, destination_folder)


zip_file = ['inputs/data_Gbif.zip',
            'inputs/data_Gbif_for_debug.zip',
            'inputs/data_RF_Gbif.zip',
            'inputs/data_RF_RSL.zip',
            'inputs/data_RLS.zip',
            'inputs/output_RF_abundace.zip',
            'inputs/outputs_cnn_sdm_presence_only.zip',
            'inputs/outputs_cnn_sdm_presence_only_for_debug.zip']
destination_folder = 'inputs'  
unzip_and_move(zip_file, destination_folder)


zip_file = ['outputs/outputs.zip']
destination_folder = 'outputs'  
unzip_and_move(zip_file, destination_folder)

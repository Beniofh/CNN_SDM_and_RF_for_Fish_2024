import zipfile
import os


def unzip_and_move(zip_files, target_dir):
    # Création du dossier cible s'il n'existe pas
    os.makedirs(target_dir, exist_ok=True)
    for zip_file_path in zip_files:
        # Décompression du fichier .zip
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
        # Suppression du fichier .zip
        #os.remove(zip_file_path)



# Chemin du dossier contenant les fichiers .zip
dossier = "./downloads"

# Liste des fichiers .zip dans le dossier
fichiers_zip_1 = [f for f in os.listdir(dossier) if f.endswith('.zip')]

# Liste des noms autorisés
noms_autorises = ['outputs.zip', 'complementary_inputs.zip', 'inputs.zip']

# Parcourt chaque fichier .zip dans le dossier
for fichier in os.listdir(dossier):
    chemin_fichier = os.path.join(dossier, fichier)
    
    # Vérifie si le fichier est un .zip et n'est pas dans la liste des noms autorisés
    if fichier.endswith('.zip') and fichier not in noms_autorises:
        print(f"{fichier} is not in the list of authorised files, check the content.")
        print("The authorised files are: 'outputs.zip', 'complementary_inputs.zip', 'inputs.zip'.")
        with zipfile.ZipFile(chemin_fichier, 'r') as archive:
            # Liste des fichiers/dossiers à la racine de l'archive
            fichiers_racine = [os.path.basename(info.filename) for info in archive.infolist() if '/' not in info.filename]

            # Vérifie si un des noms autorisés est présent à la racine
            if any(nom in fichiers_racine for nom in noms_autorises):
                print(f"{fichier} has authorised files at the root.")
                if not all(nom in fichiers_zip_1 for nom in fichiers_racine):
                    archive.extractall(dossier)
                    print(f"{fichier} has been unzipped.")
                else:
                    print(f"The authorised files at the root of {fichier} have already been unzipped.")
            else: 
                print(f"{fichier} does not have an authorised file at the root. File igniored.")

# Liste de nouveau les fichiers .zip dans le dossier
fichiers_zip_2 = [f for f in os.listdir(dossier) if f.endswith('.zip')]

if 'inputs.zip' in fichiers_zip_2:
    zip_file = ['downloads/inputs.zip']
    destination_folder = os.getcwd()  
    unzip_and_move(zip_file, destination_folder)
    print("The file 'inputs.zip' has been unzipped.")
else:
    print("The file 'inputs.zip' is missing.")

if 'outputs.zip' in fichiers_zip_2:
    zip_file = ['downloads/outputs.zip']
    destination_folder = os.getcwd() 
    unzip_and_move(zip_file, destination_folder)
    print("The file 'outputs.zip' has been unzipped.")
else:
    print("The file 'outputs.zip' is missing.")    

if 'complementary_inputs.zip' in fichiers_zip_2:
    zip_file = ['downloads/complementary_inputs.zip']
    destination_folder = 'inputs/data_Gbif'  
    unzip_and_move(zip_file, destination_folder)
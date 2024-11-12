from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import time

# Chemin vers ChromeDriver
chrome_driver_path = 'path/to/chromedriver'  # Remplacez par le chemin de votre ChromeDriver

# Configurer les options du navigateur
chrome_options = Options()
chrome_options.add_experimental_option("prefs", {
    "download.default_directory": "/path/to/download",  # Répertoire de téléchargement souhaité
    "download.prompt_for_download": False,
    "download.directory_upgrade": True,
    "safebrowsing.enabled": True
})

# Initialiser le navigateur
service = Service(chrome_driver_path)
driver = webdriver.Chrome(service=service, options=chrome_options)

try:
    # Ouvrir la page cible
    driver.get("https://entrepot.recherche.data.gouv.fr/dataset.xhtml?persistentId=doi:10.57745/O0TOF1#")  # Remplacez par l'URL réelle de la page

    # Attendre le chargement de la page
    time.sleep(5)

    # Trouver et cliquer sur le bouton de téléchargement
    download_button = driver.find_element(By.ID, "datasetForm:j_idt264")
    download_button.click()

    # Attendre le téléchargement du fichier
    time.sleep(10)  # Ajustez en fonction de la vitesse de téléchargement

finally:
    # Fermer le navigateur
    driver.quit()


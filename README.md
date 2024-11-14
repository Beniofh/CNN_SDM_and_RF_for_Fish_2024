# **Data and code for the scientific paper: "From presence-only to abundance species distribution models using transfer learning"**
 
Benjamin Bourel<sup>1*</sup>, Alexis Joly<sup>1</sup>, Maximilien Servajean<sup>2,3</sup>, Simon Bettinger<sup>4</sup>, José Antonio Sanabria Fernández<sup>5</sup>, David Mouillot<sup>4</sup>

1 Inria, University of Montpellier, LIRMM, CNRS, Montpellier, France<br>
2 IRMM, University of Montpellier, CNRS, Montpellier, France<br>
3 AMIS, Paule Valery University, Montpellier, France<br>
4 MARBEC, University of Montpellier, CNRS, IFREMER, IRD, Montpellier, France<br>
5 CRETUS - Department of Zoology, Genetics and Physical Anthropology, University of Santiago de Compostela, Santiago de Compostela, Spain<br>
\* Corresponding author: benjamin.bourel@inria.fr

**DOI of paper**: XXXXXXXXXXXXX<br>
**State of paper**: Submitted to Ecology Letter as Method paper 

## **I. Aim of this GitHug repository**
The aim of this repository is to make available the codes and data needed to reproduce the experiments in our article on species distribution models (SDM). More specifically, these experiments use SDMs based on Convolutional Neural Networks (CNN-SDMs) and  SDMs based on Random Forest (RF).

⚠️ These codes have been designed and tested under Linux (Ubuntu 22.04). Their compatibility with Window via software such as Anaconda is not guaranteed.
<br><br>

## **II. Installation** 

### **II.1. CUDA** 
The scripts for the CNN-SDMs are run on the GPU. To do this, you need to install `CUDA` if it is not already installed: [`CUDA Installation guide`](https://docs.nvidia.com/cuda/index.html) and `CuDNN Installation guide`](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html).

### **II.2. Downloading of the repository and creation of the Conda environment** 
First, you need to retrieve the GitHub repository. To do this, open the terminal in the directory where you want to place the repository, then clone the repository.  
```script
git clone git clone https://github.com/Beniofh/CNN_SDM_and_RF_for_Fish_2024
```
From the same terminal, use the following commands to go into `CNN_SDM_and_RF_for_Fish_2024` and create a `Conda` environment` using the environment.yml` file, replacing the `name_of_your_env` value with the name of your new environment. This allows you to have an environment with all the necessary pakages in the correct versions.
```script
cd CNN_SDM_and_RF_for_Fish_2024
conda env create -n name_of_your_env -f environment.yml
```
Activate your new conda environment.
```script
conda activate name_of_your_env
```
Installer `Malpolon` with `pip`. It is a modified and unofficial version of a very old version of the `Malpolon` framework. It is a framework for making CNN-SDMs that is used by the scripts at the root of scr_deep_learning_models. You can find the current version of Malpolon [`here`](https://github.com/plantnet/malpolon).
```script
pip install -e .
```
To check that the installation went well, use the following command.
```script
python -m malpolon.check_install
```
If CUDA is correctly installed, the previous command will give something similar to this.
```script
Using PyTorch version 1.13.0
CUDA available: True (version: 11.6)
cuDNN available: True (version: 8302)
Number of CUDA-compatible devices found: 1
```

### **II.3. Downloading data** 
The input and output data used in our article and needed to run the scripts are not directly on the GitHub repository because they are too large. This data is available [`here`](https://lab.plantnet.org/seafile/d/387e43c4f1ae495c96ba/). They should be available in Inria (National Institute for Research in Digital Science and Technology) open data repository in the near future.

The downloadable data consists of three .zip files :<br>

- The **`inputs.zip`** (7 Go) contains the basic input data required to run most of the scripts. These data are the environmental rasters for all fish counts, some environmental data for presence-only fish data, tables of fish species data (presence-only and counts), and the outputs of the training models used in the article (training metadata and trained model weights).

- The **`outputs.zip`** (522 Mo) contains examples of script outputs from the GitHub repository. Some of these outputs are required to run the scripts in the scr_metrics_and_figures folder in the GitHub repository.

- The **`complementary_inputs.zip`** (10 Go) contains additional input data in order to run the scripts `cnn_sdm_presence_only.py` and `pred_cnn_sdm_presence_only.py` (see part III.2) that cannot be run on standard machines. These are scripts that require a lot of computing resources and which must run on supercomputers. These additional data are all environmental rasters associated with the presence of fish. ⚠️ If you cannot or do not want to run these two scripts, you do not need to download `complementary_inputs.zip`.

### **II.4. Finalising the installation** 
After downloading the data, put the files `inputs.zip`, `outputs.zip` and `complementary_inputs.zip` in `./CNN_SDM_and_RF_for_Fish_2024/downloads` (`complementary_inputs.zip` is not obligatory as stated in part II.3). Then open a terminal in `./CNN_SDM_and_RF_for_Fish_2024` and run the commands below. This will unzip the .zip files and automatically put the files in `./CNN_SDM_and_RF_for_Fish_2024/inputs` and `./CNN_SDM_and_RF_for_Fish_2024/outputs`.
```script
conda activate name_of_your_env
python3 setup_2.py
```
<br>

## **III. Using the repository**
### **III.1. Organisation of the repository and run of the sciptes** 
The repository is organised into 6 folders:

- **`scr_deep_learning_models` folder:** It contains the scripts to be run to reproduce the deep learning experiments of article (see part III.2). More specifically, linked to the species distribution models (SDMs) based on Convolutional Neural Networks(CNN-SDMs). There are also the `modules` subfolder which contains the functions called by the scripts at the root of `scr_deep_learning_models` and the `config` subfolder which contains the .yaml configuration files for the scripts at the root of `scr_deep_learning_models`.

- **`scr_no_deep_learning_models` folder:** It contains the scripts to be run to reproduce the no deep learning experiments of article (see part III.3). More specifically, linked to the SDMs based on Random Forest (RF).

- **`scr_deep_metrics_and_figures` folder:** It contains the scripts to be run to reproduce the metrics tables and figures in the article linked to SDMs (see part III.4).

- **`malpolon` folder:** It contains a modified and unofficial version of a very old version of the `Malpolon` framework. It is a framework for making CNN-SDMs that is used by the scripts at the root of scr_deep_learning_models. You can find the current version of Malpolon [`here`](https://github.com/plantnet/malpolon).

- **outputs folder:** It contains the input data for scripts at the root of `scr_deep_learning_models`, `scr_no_deep_learning_models`, and `scr_deep_learning_models` folders.

- **inputs folder:** It contains the output data for scripts at the root of `scr_deep_learning_models`, `scr_no_deep_learning_models`, and `scr_deep_learning_models` folders.

⚠️ To run the scripts, make sure that the current directory is the scr directory corresponding to the scripts to be executed. Otherwise, relative paths in scripts such as `../outputs/abundances_results/mectric.csv` will not work. Remember also to activate the conda environment you have created. Example:
```script
(base) bbourel@aconitum:~/CNN_SDM_and_RF_for_Fish_2024$ conda activate IA-m
(IA-m) bbourel@aconitum:~/CNN_SDM_and_RF_for_Fish_2024/$ python scr_metrics_and_figures/figure_2.py 
FileNotFoundError: [Errno 2] No such file or directory: '../outputs/abundances_results/mectric.csv'
(IA-m) bbourel@aconitum:~/CNN_SDM_and_RF_for_Fish_2024$ cd scr_metrics_and_figures
(IA-m) bbourel@aconitum:~/CNN_SDM_and_RF_for_Fish_2024/scr_metrics_and_figures$ python figure_2.py 
```

### **III.2. The scripts to be run to reproduce the deep learning experiments in the articles** 
The scripts to be run to reproduce the experiments linked to the species distribution models (SDMs) based on Convolutional Neural Networks (CNN-SDMs) are the scripts at the root of the `scr_deep_learning_models` folder. The scripts whose names start with `“cnn_sdm”`, the script `pred_cnn_sdm_presence_only.py`, and the script `pred_cnn_sdm_presence_only_for_debug.py` are identical except for the variable associated with `config_name`. This variable takes the name of one of the .yaml configuration files in the `scr_deep_learning_models/config`. Depending on the .yaml file, the scripts will perform different tasks. Finally, the scripts `pred_cnn_sdm_abundance_with_tl.py` and `pred_cnn_sdm_abundance_without_tl.py` are identical except for the parameters fold_dir_path and fold_id parameters. All the scripts are described below.
<br>

  <details>
  <summary> <b>cnn_sdm_presence_only.py</b> (<i> Click here to expand description</i>)</summary>
  <div style="margin-top:10px;">

  - <u>Description:</u> It trains a CNN-SDM to predict the occurrences of 181 fish species based on 62,241 presence-only from Gbif. For each occurrence of fish, the corresponding environmental dataset is made up of 15 rasters representing 14 environmental covariates and a satellite image. Each raster is spatially centred on the GPS position of the associated occurrence. ⚠️ This script requires optional data to be downloaded (see part II.3). Running this script requires a lot of resources. It is recommended that you run it on a supercomputer, as it will not work on a personal machine.  
  - <u>Configuration file:</u> `cnn_sdm_presence_only.yaml`

  - <u>Inputs:</u> The data contained in `./inputs/data_Gbif`. The .csv file contains presence-only data (id of occurence, observation date, observed species, GPS coordinates, etc.). As stated in the paper, the presence-only data were divided into training, validation and test data sets using a random division into spatial blocks per ecoregion. The subset defined for each presence-only is shown in the .csv. Each subfolder corresponds to a type of environmental data. In each subfolder, a raster is associated with each presence-only, and the name of the raster begins with the ID of the associated presence-only.
  
  - <u>Outpus:</u> It is a folder named according to the date and time the script is run (values after seconds are miliseconds) placed in `./outputs/cnn_sdm_presence_only subfolder`. This subfolder contains .png and .csv files that can be used to view the progress of the metrics during the training. There is also a .ckpt (checkpointing) file with the weights of the model associated with its best performance, a .log file with the architecture of the model and a .yaml file with the configuration used. Note that the script automatically replaces ‘auto’ values with calculated values and converts relative paths into absolute paths. This .yaml file shows the configuration after this automation step. To see the real original configuration, go to the `.hydra` subfolder and look at the `config.yaml` file.
  </div>
  </details>
<br>

  <details>
  <summary> <b>cnn_sdm_presence_only_for_debug.py</b> (<i>Click here to expand description</i>)</summary>
  <div style="margin-top:10px;">
  
  - <u>Description:</u> It does the same thing as the cnn_sdm_presence_only.py script except that it uses a very reduced version of the Gbif dataset (96 presence of only 32 fish species). This dataset is not sufficient to obtain suitable results, but it allows you to see how the cnn_sdm_presence_only.py script works without needing a supercomputer.
  
  - <u>Configuration file:</u> `cnn_sdm_presence_only_for_debug.yaml`. This is the same as `cnn_sdm_presence_only.yaml` but the path names data.dataset_path and data.csv_occurence_path have been changed.
  
  - <u>Inputs:</u> The data in `inputs/data_Gbif_for_debug`. Inputs are organised as for `cnn_sdm_presence_only.py`.

  - <u>Outpus:</u> The same type as `cnn_sdm_presence_only.py` but located in `./outputs/cnn_sdm_presence_only_for_debug`.
  </div>
  </details>
<br>
  
  <details>
  <summary> <b>cnn_sdm_abundance_without_tl.py</b> (<i>Click here to expand description</i>)</summary>
  <div style="margin-top:10px;">
  
  - <u>Description:</u> It is used to train a CNN-SDM for predicting species abundance based on 406 fish counts representing 47 species. The count data comes from the database of the Reef Life Survey program carried out in the Mediterranean Sea between 2011 and 2020. For each fish count, the corresponding environmental dataset is made up of 15 rasters representing 14 environmental covariates and a satellite image (the same as for occurrence models). Each raster is spatially centred on the GPS position of the associated fish count. As stated in the article, the results of CNN-SDMs for fish abundances are based on a k-fold cross-validation, using k=20 random spatial block splits from the dataset of fish abundances. The default setting for this script is to reproduce the results for flod n°0. You can run the script for the other folds by modifying the `transfer_learning.data_tf.csv_occurence_path` value in the configuration file. For example, to reproduce the results for fold 12, replace ../inputs/data_RLS/Galaxy117-Sort_on_data_82_n_vec_subset_0.csv with ../inputs/data_RLS/Galaxy117-Sort_on_data_82_n_vec_subset_12.csv.
  
  - <u>Configuration file:</u> `cnn_sdm_abundance_without_tl.yaml`. This is the same as cnn_sdm_presence_only.yaml but with transfer_learning.transfer_learning_activated = True. This allows the model to be trained to predict abundance but without recovering the weight of the pre-trained model based on presence-only, because transfer_learning.load_checkpoint = False. 
  
  - <u>Inputs:</u> The input data can be found in two folders.
    - The data for training the model to predict fish abundance values are in the `./inputs/data_RLS folder` and are organised as for cnn_sdm_abundance_without_tl.py. The only difference is that there is not one but 20 .csv files. These 20 .csv files contain the same information except for the subset (train, val, or test) associated with each count. These 20 .csv files are numbered from 0 to 19 and correspond respectively to the flods 0 to 19 mentioned in the article
    - The data used to retrieve information about the data used to train the occurrence model in order to adapt the new model is found in `./inputs/data_Gbif`.  This includes the .csv file and the 15 rasters (one in each subfolder) associated with occurrence no. 3853246815.
    
  - <u>Outpus:</u> The same type as cnn_sdm_presence_only.py but located in `./outputs/cnn_sdm_abundance_without_tl`. 
  </div>
  </details>
<br>

  <details>
  <summary> <b>cnn_sdm_abundance_with_tl.py</b> (<i>Click here to expand description</i>)</summary>
  <div style="margin-top:10px;">
  
  - <u>Description:</u> As for `cnn_sdm_abundance_without_tl.py`, but applying transfer learning from a model trained on presence data only with the script `cnn_sdm_presence_only.py`.

  
  - <u>Configuration file:</u> `cnn_sdm_abundance_with_tl.yaml`. This is the same as `cnn_sdm_presence_only.yaml` but with transfer_learning.transfer_learning_activated = True and transfer_learning.load_checkpoint = True.
  
  - <u>Inputs:</u> The input data can be found in three folders.
    - The data for training the model to predict fish abundance values are in the `./inputs/data_RLS folder and are organised` as for cnn_sdm_abundance_without_tl.py. The only difference is that there is not one but 20 .csv files. These 20 .csv files contain the same information except for the subset (train, val, or test) associated with each count. These 20 .csv files are numbered from 0 to 19 and correspond respectively to the flods 0 to 19 mentioned in the article.
    - The data used to retrieve information about the data used to train the occurrence model in order to adapt the new model is found in `./inputs/data_Gbif.` This includes the .csv file and the 15 rasters (one in each subfolder) associated with occurrence no. 3853246815.
-outpus: of the same type as cnn_sdm_presence_only.py but located in `./outputs/cnn_sdm_abundance_without_tl`.
    - The file checkpoint-epoch=09-step=11400-val_accuracy=0.0804.ckpt containing the weights of the model trained to predict occurrences and used for transfer learning can be found in the folder `./inputs/outputs_cnn_sdm_presence_only/2023-10-06_16-38_085373`. This folder `2023-10-06_16-38_085373` is the output of cnn_sdm_presence_only.py which was used in the article. 
  
  - <u>Outpus:</u> The same type as `cnn_sdm_presence_only.py` but located in `./outputs/cnn_sdm_abundance_with_tl`.
  </div>
  </details>
<br>

  <details>
  <summary> <b>pred_cnn_sdm_presence_only.py</b> (<i>Click here to expand description</i>)</summary>
  <div style="margin-top:10px;">
  
  - <u>Description:</u> It is used to calculate the metic values for the validation and test sets for a CNN-SDM trained in presence-only with cnn_sdm_presence_only.py. ⚠️This script requires optional data to be downloaded (see part II.3). Running this script requires a lot of resources. It is recommended that you run it on a supercomputer, as it will not work on a personal machine.
  
  - <u>Configuration file: </u>`pred_cnn_sdm_presence_only.yaml`. This is the same as `cnn_sdm_presence_only.yaml` but with visualization.validate_metric = True.
  
  - <u>Inputs:</u> The input data can be found in two folders.
    -The data contained in `./inputs/data_Gbif`, as `cnn_sdm_presence_only.py`.
    -The file `checkpoint-epoch=09-step=11400-val_accuracy=0.0804.ckpt` containing the weights of the model trained to predict occurrences can be found in the folder `./inputs/outputs_cnn_sdm_presence_only/2023-10-06_16-38_085373`. This folder `2023-10-06_16-38_085373` is the output of `cnn_sdm_presence_only.py` which was used in the article. 
  
  - <u>Outpus:</u> The metic values are printed directly to the terminal.
  </div>
  </details>
<br>

  <details>
  <summary> <b>pred_cnn_sdm_presence_only_for_debug.py</b> (<i>Click here to expand description</i>)</summary>
  <div style="margin-top:10px;">
  
  - <u>Description:</u> It is used to calculate the metic values for the validation and test sets for a CNN-SDM trained in presence-only with `cnn_sdm_presence_only_for_debug.py`.

  
  - <u>Configuration file:</u>`pred_cnn_sdm_presence_only.yaml`. This is the same as cnn_sdm_presence_only.yaml but with visualization.validate_metric = True and the value of visualization.chk_path_validate_metric has been modified.
  
  - <u>Inputs:</u> The input data can be found in two folders.
    - The data contained in `inputs/data_Gbif_for_debug`, as `cnn_sdm_presence_only_for_debug.py`
    - The file `checkpoint-epoch=00-step=1-val_accuracy=0.0667.ckpt` containing the weights of the model trained to predict occurrences can be found in the folder `./inputs/outputs_cnn_sdm_presence_only_for_debug/22024-11-06_12-30-28_624128`. This folder `2024-11-06_12-30-28_624128` is a output of cnn_sdm_presence_only_for_debug.py. 

  - <u>Outpus:</u> The metic values are printed directly to the terminal.
  </div>
  </details>
 <br>

  <details>
  <summary> <b>pred_cnn_sdm_abundance_without_tl.py</b> (<i>Click here to expand description</i>)</summary>
  <div style="margin-top:10px;">
  
  - <u>Description:</u> It uses a CNN-SDM trained, with `cnn_sdm_abundance_without_tl.py`, to predict abundance without transfer learning to predict the abundance of 47 poission species for each site. 
  
  - <u>Configuration file:</u>`pred_cnn_sdm_abundance.yaml`. This configuration file only specifies the location for the outputs. The script directly retrieves the configurations saved in the output folders of `cnn_sdm_abundance_without_tl.py` in `.hydra/config.yaml`.
  
  - <u>Inputs:</u> The input data can be found in two folders.
    - The data contained in `inputs/data_RLS`, as `cnn_sdm_abundance_without_tl.py`.
    - All the data contained in `inputs/data_Gbif`.
    -The folders in `./inputs/outputs_cnn_sdm_abundance_without_tl`. Each of these 20 folders is the output of `cnn_sdm_abundance_without_tl.py` for each of the 20 folds in the article. These are the data used in the article. In each folder, the configuration used to train the model is retrieved in the `./.hydra/config.yaml` file and the weights of the trained model in the .ckpt file. These folders are accompanied by a .dos metadata file indicating which fold of the article each folder corresponds to (this .ods file has been completed manually and is not used as input). 

  - <u>Outpus:</u> The files in `./outputs/pred_cnn_sdm_abundance_without_tl`, 20 .csv files with predictions per site and 20 .csv files where the predictions are staked. These 20 .csv files are numbered from 0 to 19 and correspond respectively to the flods 0 to 19 mentioned in the article.
  </div>
  </details>
<br>
  
  <details>
  <summary> <b>pred_cnn_sdm_abundance_with_tl.py</b> (<i>Click here to expand description</i>)</summary>
  <div style="margin-top:10px;">

  - <u>Description:</u> It uses a CNN-SDM trained, with `cnn_sdm_abundance_without_tl.py`, to predict abundance with transfer learning to predict the abundance of 47 poission species for each site.
  
  - <u>Configuration file:</u> As `pred_cnn_sdm_abundance_without_tl.py`.  
  
  - <u>Inputs:</u> As pred_cnn_sdm_abundance_without_tl.py but  in `./inputs/outputs_cnn_sdm_abundance_with_tl`.
  
  - <u>Outpus:</u> As pred_cnn_sdm_abundance_without_tl.py but in `./outputs/pred_cnn_sdm_abundance_with_tl`.
  </div>
  </details>
<br>

### **III.3. The scripts to be run to reproduce the deep learning experiments in the articles**
The scripts to be run to reproduce the experiments linked to the species distribution models (SDMs) based on Random Forest (RF) are the scripts at the root of the `scr_no_deep_learning_models` folder. All the scripts are described below.
<br>

  <details>
  <summary> <b>pred_RF_presence-only.py</b> (<i>Click here to expand description</i>)</summary>
  <div style="margin-top:10px;">
  
  - <u>Description:</u> It trains a CNN-SDM to predict the occurrences of 181 fish species based on 62,241 presence-only from Gbif. For each occurrence of fish, the corresponding environmental dataset is made up of 15 rasters representing 14 environmental covariates and a satellite image. RFs cannot deal directly with the rasterised environmental data as the CNN-SDM does (this leads to strong over-fitting). To solve this problem, we use the mean of the pixels and the standard deviation of each raster used with sdm_abundance_presence_only.py.
  
  - <u>Inputs:</u> The input data are .csv in `./inputs/data_RF_Gbif`. These are lists of occurrences with environmental data in one file and the split between training, validation and test subsets in the second. 

  - <u>Outpus:</u> Top 1, 5, 10 and 20 micro and macro accuracy values for the train, validation and test set printed directly to the terminal.
  </div>
  </details>
<br>

  <details>
  <summary> <b>pred_RF_abundace.py</b> (<i>Click here to expand description</i>)</summary>
  <div style="margin-top:10px;">

  - <u>Description:</u> It is used to train RF models for predicting species abundance based on 406 fish counts representing 47 species. The count data comes from the database of the Reef Life Survey program carried out in the Mediterranean Sea between 2011 and 2020. RFs cannot deal directly with the rasterised environmental data as the CNN-SDM does (this leads to strong over-fitting). To solve this problem, we use the mean of the pixels and the standard deviation of each raster used with sdm_abundance_without_tl.py. This script does this 20 times, once for each fold in the article.
  
  - <u>Inputs:</u> The input data can be found in two folders.
    - The file `Galaxy117-Sort_on_data_82_n_vec_value.csv` in `./inputs/data_RF_RSL` which contains the fish counts per site and the values of the environmental variables for each site.
    - The 20 .csv files in `./inputs/data_RLS`. These 20 .csv files contain the same information except for the subset (train, val, or test) associated with each count. These 20 .csv files are numbered from 0 to 19 and correspond respectively to the flods 0 to 19 mentioned in the article.  

  - <u>Outpus:</u> The files in `./outputs/pred_RF_abundace`, 20 .csv files with predictions per site and 20 .csv files where the predictions are staked. These 20 .csv files are numbered from 0 to 19 and correspond respectively to the flods 0 to 19 mentioned in the article. For each fold, there is also the mean decrease in impurity and the permutation importances visible via .png files. Finally, there is the config.py file, which is a copy of the configuration used.
  </div>
  </details>
<br>

### **III.4. The scripts to be run to reproduce tables and figures**
The scripts to be run to reproduce the metrics tables and figures in the article linked to SDMs are the scripts at the root of the `scr_metrics_and_figures` folder.. All the scripts are described below.
<br>

  <details>
  <summary> <b>metric.py</b> (<i>Click here to expand description</i>)</summary>
  <div style="margin-top:10px;">

  - <u>Description:</u> For each of the three abundance models presented in the article (CNN-SDM without transfer learning, CNN-SDM with transfer learning and RF), it calculates the values of the three metrics presented in the article (D-squared regression score function on the log-transformed data, the Spearman rank-order coefficient, and the R-squared regression score function on Log-transformed data) on the test data for the 20 training cycles of the model associated with each of the 20 folds.
  
  - <u>Inputs:</u> The set of .csv files in `./outputs/pred_cnn_sdm_abundance_with_tl`, `./outputs/pred_cnn_sdm_abundance_with_tl`, and `./outputs/pred_RF_abundace`. These correspond to the outputs of `pred_cnn_sdm_abundance_without_tl.py`,`pred_cnn_sdm_abundance_with_tl.py`, and `pred_RF_abundace.py` scripts respectively.

  - <u>Outpus:</u> The `metric.csv` and `metric_by_sp.csv` files in `./outputs/abundances_results`. `metric.csv` shows the values of the metrics for each flod for the 3 models. `metric_by_sp.csv` shows the same thing but goes into more detail by giving the values for each species. 
  </div>
  </details>
<br>

  <details>
  <summary> <b>figure_2.py</b> (<i>Click here to expand description</i>)</summary>
  <div style="margin-top:10px;">

  - <u>Description:</u> It reproduces Figure 2 of the article. Violin plots showing the model performances on the fish abundance test set over the 20 folds for (A) the D-squared regression score function on the log-transformed data (D2log), (B) the Spearman rank-order coefficient (Spearman coefficient) and (C) the R-squared regression score function on Log-transformed data (R2log). For these three metrics, the closer the value is to 1, the better the model. Std = Standard deviation.
  
  - <u>Inputs:</u> The `mectric.csv` file in `./outputs/abundances_results/`. This file is one of the outputs of `metric.py`.

  - <u>Outpus:</u> A .jpeg file with a name starting with `figure_2` in `./outputs/abundances_results`.
  </div>
  </details>
<br>

  <details>
  <summary> <b>figure_3.py</b> (<i>Click here to expand description</i>)</summary>
  <div style="margin-top:10px;">

  - <u>Description:</u> It reproduces Figure 3 of the article. D-squared regression score function on the log-transformed data (D2Log) deviation by species between CNN-SDM with transfer learning and Random Forest calculated for each of the 20 folds. The percentage in brackets next to the name of each species indicates the percentage of the site studied on which they are present.
  
  - <u>Inputs:</u> The `mectric.csv` file in `./outputs/abundances_results/` that is a output of `metric.py`. The file `Galaxy117-Sort_on_data_82_n.csv` in `./inputs/data_RF_RSL` which contains the fish counts per site.

  - <u>Outpus:</u> A .jpeg file with a name starting with `figure_3` in `./outputs/abundances_results`.
  </div>
  </details>
<br>

## **Acknowledgments**
This work has been mainly funded by the IA-Biodiv ANR project FISH-PREDICT (ANR-21-AAFI-0001-01). It was also partially funded by the European Union’s Horizon research and innovation program under grant agreement No 101060639 (MAMBO project) and No 101060693 (GUARDEN project). This work was granted access to the HPC resources of IDRIS under the allocation 2022- AD011013891 made by GENCI ».
 
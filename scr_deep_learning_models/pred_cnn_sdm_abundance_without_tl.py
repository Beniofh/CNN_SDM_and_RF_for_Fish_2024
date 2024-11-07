import os
import shutil
import glob

import hydra
import torch
from omegaconf import DictConfig
from omegaconf import OmegaConf

import pandas as pd
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from malpolon.logging import Summary

from modules.transforms import *
from modules.auto_plot import Autoplot

from modules.init_elements import Init_of_secondary_parameters

from pytorch_lightning.callbacks import LearningRateMonitor

from modules.transfer_learning import Transfer_learning_ia_biodiv
from modules.datamodule import MicroGeoLifeCLEF2022DataModule, ClassificationSystem

import yaml
   
import torch


@hydra.main(version_base="1.1", config_path="config", config_name="pred_cnn_sdm_abundance")
def main(cfg: DictConfig) -> None:
    logger = pl.loggers.CSVLogger(".", name=False, version="")
    logger.log_hyperparams(cfg)
    
    repertoire_courant_original = os.getcwd()
    os.chdir(os.path.dirname(__file__))

    #Relu without_transfer_learning        
    fold_dir_path = os.path.abspath("../inputs/outputs_cnn_sdm_abundance_without_tl")+ "/"
    fold_id = ['2023-10-09_15-52-40_554053','2023-10-06_12-00-30_289910','2023-10-06_12-22-11_674802','2023-10-09_13-51-50_793981',
               '2023-11-14_14-07-22_792849','2023-11-14_11-24-38_478298','2023-11-14_11-51-24_985452','2023-11-14_12-01-08_328114',
               '2023-11-14_13-01-46_067539','2023-11-14_13-07-38_628403','2024-03-15_14-56-14_998228','2024-03-15_15-15-33_299152',
               '2024-03-15_15-29-10_776701','2024-03-15_15-55-35_626984','2024-03-15_16-06-19_134325','2024-03-18_10-42-55_711583',
               '2024-03-15_16-40-12_062533','2024-03-15_17-43-07_616188','2024-03-18_10-22-25_584030','2024-03-15_18-05-43_296206'] 

    os.chdir(repertoire_courant_original)

    for fold, flod_number in zip(fold_id, range(0, len(fold_id))):
        fold_path = fold_dir_path + fold
        chk_path = glob.glob(fold_path + '/*.ckpt')[-1]
        path_chk=Path(chk_path)
        
        path_yaml=path_chk.parent / ".hydra/config.yaml"
        with open(path_yaml, 'r') as file:
            cfg = yaml.safe_load(file)
        cfg = OmegaConf.create(cfg)

        os.chdir(os.path.dirname(__file__))
        cfg.data.dataset_path = os.path.abspath(cfg.data.dataset_path)
        cfg.data.csv_occurence_path = os.path.abspath(cfg.data.csv_occurence_path)
        cfg.visualization.chk_path_validate_metric = os.path.abspath(cfg.visualization.chk_path_validate_metric)
        cfg.transfer_learning.chk_path = os.path.abspath(cfg.transfer_learning.chk_path)
        cfg.transfer_learning.model_tf.chk_tf_path = os.path.abspath(cfg.transfer_learning.model_tf.chk_tf_path)
        cfg.transfer_learning.data_tf.dataset_path = os.path.abspath(cfg.transfer_learning.data_tf.dataset_path)
        cfg.transfer_learning.data_tf.csv_occurence_path = os.path.abspath(cfg.transfer_learning.data_tf.csv_occurence_path)
        os.chdir(repertoire_courant_original)

        cls_num_list_train, patch_data_ext, cfg, cfg_modif, patch_band = Init_of_secondary_parameters(cfg=cfg)
            
        datamodule = MicroGeoLifeCLEF2022DataModule(**cfg.data,
                                                    patch_data_ext = patch_data_ext,
                                                    patch_data=cfg.patch.patch_data, 
                                                    patch_band_mean = cfg.patch.patch_band_mean,
                                                    patch_band_sd = cfg.patch.patch_band_sd,
                                                    train_augmentation = cfg.train_augmentation,
                                                    test_augmentation = cfg.test_augmentation,
                                                    dataloader = cfg.dataloader, )

        
        cfg_model = hydra.utils.instantiate(cfg_modif.model)
        
        model = ClassificationSystem(cfg_model, **cfg.optimizer.SGD, **cfg.optimizer.scheduler, **cfg.optimizer.loss, dropout_proba = cfg.dropout_proba, cls_num_list_train=cls_num_list_train)

        
        callbacks = [
            Summary(),
            ModelCheckpoint(
                dirpath=os.getcwd(),
                filename="checkpoint-{epoch:02d}-{step}-{val_accuracy:.4f}",
                monitor= cfg.callbacks.monitor,
                mode=cfg.callbacks.mode,),
            LearningRateMonitor(logging_interval=cfg.optimizer.scheduler.logging_interval), #epoch'),
            EarlyStopping(monitor=cfg.callbacks.monitor, min_delta=0.00, patience=cfg.callbacks.patience, verbose=False, mode=cfg.callbacks.mode)]                
                
        trainer = pl.Trainer(logger=logger, callbacks=callbacks, **cfg.trainer)

        model, datamodule, trainer = Transfer_learning_ia_biodiv(model, cfg, cfg_model, cls_num_list_train, patch_data_ext, logger)
        
        checkpoint = torch.load(chk_path) 
        model.load_state_dict(checkpoint['state_dict'])
        
        datamodule.setup(stage="fit")

        path_input = datamodule.csv_occurence_path
        path_input_temp = datamodule.csv_occurence_path[:-4] + '_temp.csv'
        
        input = pd.read_csv(path_input)
        input_temp = input.copy()
        input_temp["subset"]="test"
        input_temp.to_csv(path_input_temp, index=False)
        datamodule.csv_occurence_path = path_input_temp
        trainer.test(model, datamodule=datamodule)
        
        os.remove(path_input_temp)        
        input[datamodule.csv_col_occurence_id][input["subset"]=="test"]
        
        y_pred = model.y_pred
        y_true = model.y_true
        
        if cfg_modif.transfer_learning.model_tf.num_outputs_tf ==47 :
            list_sp = ['Apogon imberbis', 'Atherina hepsetus', 'Boops boops', 'Centrolabrus exoletus', 'Chelon labrosus', 'Chromis chromis', 'Chrysophrys auratus', 'Coris julis', 'Ctenolabrus rupestris', 'Dentex dentex', 'Diplodus annularis', 'Diplodus cervinus', 'Diplodus puntazzo', 'Diplodus sargus', 'Diplodus vulgaris', 'Epinephelus costae', 'Epinephelus marginatus', 'Gobius bucchichi', 'Gobius xanthocephalus', 'Labrus merula', 'Labrus viridis', 'Mullus surmuletus', 'Muraena helena', 'Oblada melanura', 'Octopus vulgaris', 'Pagrus pagrus', 'Parablennius pilicornis', 'Parablennius rouxi', 'Pomadasys incisus', 'Sarpa salpa', 'Sciaena umbra', 'Seriola dumerili', 'Serranus cabrilla', 'Serranus scriba', 'Spicara maena', 'Spicara smaris', 'Spondyliosoma cantharus', 'Symphodus cinereus', 'Symphodus doderleini', 'Symphodus mediterraneus', 'Symphodus melanocercus', 'Symphodus ocellatus', 'Symphodus roissali', 'Symphodus rostratus', 'Symphodus tinca', 'Thalassoma pavo', 'Tripterygion delaisi']
        elif cfg_modif.transfer_learning.model_tf.num_outputs_tf == 49 :
            list_sp = ['Anthias anthias', 'Apogon imberbis', 'Atherina hepsetus', 'Boops boops', 'Chelon labrosus', 'Chromis chromis', 'Chrysophrys auratus', 'Coris julis', 'Ctenolabrus rupestris', 'Dentex dentex', 'Diplodus annularis', 'Diplodus cervinus', 'Diplodus puntazzo', 'Diplodus sargus', 'Diplodus vulgaris', 'Epinephelus costae', 'Epinephelus marginatus', 'Gobius bucchichi', 'Gobius xanthocephalus', 'Labrus merula', 'Labrus viridis', 'Mugil cephalus', 'Mullus surmuletus', 'Muraena helena', 'Oblada melanura', 'Pagrus pagrus', 'Parablennius pilicornis', 'Parablennius rouxi', 'Pomadasys incisus', 'Sarpa salpa', 'Sciaena umbra', 'Scorpaena notata', 'Seriola dumerili', 'Serranus cabrilla', 'Serranus scriba', 'Sphyraena viridensis', 'Spicara maena', 'Spicara smaris', 'Spondyliosoma cantharus', 'Symphodus cinereus', 'Symphodus doderleini', 'Symphodus mediterraneus', 'Symphodus melanocercus', 'Symphodus ocellatus', 'Symphodus roissali', 'Symphodus rostratus', 'Symphodus tinca', 'Thalassoma pavo', 'Tripterygion delaisi']

        df_pred = pd.DataFrame(y_pred)
        df_true = pd.DataFrame(y_true)

        df_pred = df_pred.rename(columns=dict(zip(df_pred.columns, list_sp)))
        df_true = df_true.rename(columns=dict(zip(df_true.columns, list_sp)))

        df_pred.insert(0,datamodule.csv_col_occurence_id ,input[datamodule.csv_col_occurence_id])
        df_pred.insert(1, "subset", input["subset"])
        df_pred.insert(2, "type", ['pred_{}'.format(i+1) for i in range(len(df_pred))])
        df_true.insert(0,datamodule.csv_col_occurence_id ,input[datamodule.csv_col_occurence_id])
        df_true.insert(1, "subset", input["subset"])
        df_true.insert(2, "type", ['true_{}'.format(i+1) for i in range(len(df_pred))])
        
        df_final = pd.concat([df_pred, df_true], ignore_index=True)

        df_pred_test_stack = pd.DataFrame(df_pred.iloc[:, 3:][df_pred["subset"]=="test"].stack().reset_index(drop=True))
        df_pred_test_stack.insert(0, "subset", "test")
        df_pred_test_stack.insert(0, "specie", list_sp*df_pred[df_pred["subset"]=="test"].shape[0])
        df_pred_test_stack.insert(0, "SurveyID", [item for item in df_pred["SurveyID"][df_pred["subset"]=="test"].tolist() for _ in range(len(list_sp))])

        df_pred_val_stack = pd.DataFrame(df_pred.iloc[:, 3:][df_pred["subset"]=="val"].stack().reset_index(drop=True))
        df_pred_val_stack.insert(0, "subset", "val")
        df_pred_val_stack.insert(0, "specie", list_sp*df_pred[df_pred["subset"]=="val"].shape[0])
        df_pred_val_stack.insert(0, "SurveyID", [item for item in df_pred["SurveyID"][df_pred["subset"]=="val"].tolist() for _ in range(len(list_sp))])

        df_pred_train_stack = pd.DataFrame(df_pred.iloc[:, 3:][df_pred["subset"]=="train"].stack().reset_index(drop=True))
        df_pred_train_stack.insert(0, "subset", "train")
        df_pred_train_stack.insert(0, "specie", list_sp*df_pred[df_pred["subset"]=="train"].shape[0])
        df_pred_train_stack.insert(0, "SurveyID", [item for item in df_pred["SurveyID"][df_pred["subset"]=="train"].tolist() for _ in range(len(list_sp))])

        df_true_test_stack = df_true.iloc[:, 3:][df_true["subset"]=="test"].stack().reset_index(drop=True)
        df_true_val_stack = df_true.iloc[:, 3:][df_true["subset"]=="val"].stack().reset_index(drop=True)
        df_true_train_stack = df_true.iloc[:, 3:][df_true["subset"]=="train"].stack().reset_index(drop=True)

        df_final_stack = pd.concat(
            [pd.concat([df_pred_test_stack, df_pred_val_stack, df_pred_train_stack], ignore_index=True),
            pd.concat([df_true_test_stack, df_true_val_stack, df_true_train_stack], ignore_index=True)],
            axis=1)
   
        df_final_stack.columns=["SurveyID", "specie", "subset","pred", "true"]
        
        df_final.to_csv(os.getcwd()[:-len(Path(os.getcwd()).name)] + "pred_flod_" + str(flod_number) + '.csv', index=False)
        df_final_stack.to_csv(os.getcwd()[:-len(Path(os.getcwd()).name)] + "pred_flod_" + str(flod_number) + '_stacked.csv', index=False)
    shutil.rmtree(os.getcwd())

if __name__ == "__main__":
        main()
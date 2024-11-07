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

from transforms import *
from auto_plot import Autoplot

from init_elements import Init_of_secondary_parameters

from pytorch_lightning.callbacks import LearningRateMonitor

from transfer_learning import Transfer_learning_ia_biodiv
from auto_lr_finder import Auto_lr_find
from datamodule import MicroGeoLifeCLEF2022DataModule, ClassificationSystem

import yaml
   
import torch


@hydra.main(version_base="1.1", config_path="config", config_name="cnn_multi_band_config")
def main(cfg: DictConfig) -> None:
    logger = pl.loggers.CSVLogger(".", name=False, version="")
    logger.log_hyperparams(cfg)

    # with_transfer_learning  
    fold_dir_path = "/home/bbourel/Documents/2_Travaux_scientifiques_en_cours/2024_Fish_Predict_(Bourel_et_al)/Archive_output_malpolon/Exp_RLS_local/without_relu/with_transfer_learning/"
    fold_id = ["2024-06-26_14-29-31_300111", "2024-06-24_13-46-19_778518", "2024-06-26_15-21-35_515990", "2024-06-24_14-13-56_081209",
               "2024-06-24_14-32-28_846179", "2024-06-26_10-07-03_320483", "2024-06-28_13-26-14_774893", "2024-06-24_15-54-38_223621",
               "2024-06-24_16-08-02_400420", "2024-06-26_10-36-22_400500", "2024-06-24_17-06-01_627116", "2024-06-25_09-47-40_329036",
               "2024-06-25_10-08-02_666761", "2024-06-25_12-15-44_498904", "2024-06-25_15-23-20_313803", "2024-06-25_16-20-03_496385",
               "2024-06-25_16-50-38_393124", "2024-06-26_11-11-24_734444", "2024-06-25_17-13-15_230227", "2024-06-26_09-34-03_691279"]
               
    '''
    #Relu with_transfer_learning    
    fold_dir_path = "/home/bbourel/Documents/2_Travaux_scientifiques_en_cours/2024_Fish_Predict_(Bourel_et_al)/Archive_output_malpolon/Exp_RLS_local/with_relu/"
    fold_id = ['2023-10-10_10-39-42_062309','2023-10-10_10-51-37_044027','2023-10-10_11-04-12_858569','2023-10-10_13-43-18_799765',
           '2023-11-14_10-57-46_066774','2023-11-14_11-28-57_728295','2023-11-14_11-41-08_942463','2023-11-14_12-44-38_742345',
           '2023-11-14_12-50-30_413615','2023-11-14_13-40-10_757809','2024-03-15_15-04-09_573918','2024-03-18_10-31-15_941707',
           '2024-03-15_15-37-32_496173','2024-03-15_16-01-20_658813','2024-03-15_16-12-39_444462','2024-03-18_10-48-38_335539',
           '2024-03-15_17-06-39_906401','2024-03-15_17-49-11_269416','2024-03-15_17-59-45_820416','2024-03-15_18-11-15_943220']

    #Relu without_transfer_learning        
    fold_dir_path = "/home/bbourel/Documents/2_Travaux_scientifiques_en_cours/2024_Fish_Predict_(Bourel_et_al)/Archive_output_malpolon/Exp_RLS_local/with_relu/"
    fold_id = ['2023-10-09_15-52-40_554053','2023-10-06_12-00-30_289910','2023-10-06_12-22-11_674802','2023-10-09_13-51-50_793981',
           '2023-11-14_14-07-22_792849','2023-11-14_11-24-38_478298','2023-11-14_11-51-24_985452','2023-11-14_12-01-08_328114',
           '2023-11-14_13-01-46_067539','2023-11-14_13-07-38_628403','2024-03-15_14-56-14_998228','2024-03-15_15-15-33_299152',
           '2024-03-15_15-29-10_776701','2024-03-15_15-55-35_626984','2024-03-15_16-06-19_134325','2024-03-18_10-42-55_711583',
           '2024-03-15_16-40-12_062533','2024-03-15_17-43-07_616188','2024-03-18_10-22-25_584030','2024-03-15_18-05-43_296206'] 
    
    #without_transfer_learning    
    fold_dir_path = "/home/bbourel/Documents/2_Travaux_scientifiques_en_cours/2024_Fish_Predict_(Bourel_et_al)/Archive_output_malpolon/Exp_RLS_local/without_relu/without_transfer_learning/"
    fold_id = ["2024-06-28_14-14-49_328510", "2024-06-28_14-50-52_317873", "2024-06-28_15-10-10_177613", "2024-06-28_15-23-28_925717",
               "2024-06-28_15-44-01_139889", "2024-06-28_16-04-44_481878", "2024-06-28_16-07-41_765973", "2024-06-28_16-39-34_228146",
               "2024-06-28_16-53-49_955318", "2024-06-28_17-23-42_882601", "2024-06-28_17-41-10_568525", "2024-06-28_18-23-29_345073",
               "2024-07-01_10-09-20_882625", "2024-07-01_10-44-25_883034", "2024-07-01_11-09-29_201687", "2024-07-01_11-39-08_597973",
               "2024-07-01_11-47-14_650928", "2024-07-01_12-10-13_337866", "2024-07-01_12-20-07_966731", "2024-07-01_13-35-54_843810"]
    # with_transfer_learning  
    fold_dir_path = "/home/bbourel/Documents/2_Travaux_scientifiques_en_cours/2024_Fish_Predict_(Bourel_et_al)/Archive_output_malpolon/Exp_RLS_local/without_relu/with_transfer_learning/"
    fold_id = ["2024-06-26_14-29-31_300111", "2024-06-24_13-46-19_778518", "2024-06-26_15-21-35_515990", "2024-06-24_14-13-56_081209",
               "2024-06-24_14-32-28_846179", "2024-06-26_10-07-03_320483", "2024-06-28_13-26-14_774893", "2024-06-24_15-54-38_223621",
               "2024-06-24_16-08-02_400420", "2024-06-26_10-36-22_400500", "2024-06-24_17-06-01_627116", "2024-06-25_09-47-40_329036",
               "2024-06-25_10-08-02_666761", "2024-06-25_12-15-44_498904", "2024-06-25_15-23-20_313803", "2024-06-25_16-20-03_496385",
               "2024-06-25_16-50-38_393124", "2024-06-26_11-11-24_734444", "2024-06-25_17-13-15_230227", "2024-06-26_09-34-03_691279"]
    
    fold_dir_path = "/home/bbourel/Documents/IA/malpolon/examples_perso/multiband_unimodel/outputs/cnn_multi_band_en_routine/"
    fold_id = ['2024-07-01_13-35-54_843810']
    '''

    for fold, flod_number in zip(fold_id, range(0, len(fold_id))):
        fold_path = fold_dir_path + fold

        chk_path = glob.glob(fold_path + '/*.ckpt')[-1]
    
        path_chk=Path(chk_path)
        path_yaml=path_chk.parent / "hparams.yaml"
        with open(path_yaml, 'r') as file:
            cfg = yaml.safe_load(file)
        cfg=OmegaConf.create(cfg)

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
        
        #test
        #####
        import torch.nn as nn
        model.model = nn.Sequential(*list(model.model.children())[:-1])
        #####

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
        
        #test
        #####
        y_pred = y_pred.squeeze(axis=2).squeeze(axis=2)
        #####
        
        df_pred = pd.DataFrame(y_pred)
        df_pred.insert(0,datamodule.csv_col_occurence_id ,input[datamodule.csv_col_occurence_id])
        df_pred.insert(1, "subset", input["subset"])
        df_pred.insert(2, "type", ['pred_{}'.format(i+1) for i in range(len(df_pred))])
        
        chk_name = chk_path.split(os.sep)[-2]

        
        df_pred.to_csv(os.getcwd()[:-len(Path(os.getcwd()).name)] + "feature_vector_flod_" + str(flod_number) + '.csv', index=False)
    shutil.rmtree(os.getcwd())

if __name__ == "__main__":
        main()
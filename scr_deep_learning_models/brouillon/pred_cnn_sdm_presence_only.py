import os
import shutil
import glob
import sys

import hydra
import torch
from omegaconf import DictConfig
from omegaconf import OmegaConf
import pandas as pd
from datetime import datetime
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
from modules.auto_lr_finder import Auto_lr_find
from modules.datamodule import MicroGeoLifeCLEF2022DataModule, ClassificationSystem



@hydra.main(version_base="1.1", config_path="config", config_name="cnn_sdm_presence_only_for_debug")
def main(cfg: DictConfig) -> None:
    import yaml
    logger = pl.loggers.CSVLogger(".", name=False, version="")
    logger.log_hyperparams(cfg)
    
    fold_dir_path = os.path.dirname(os.path.dirname(__file__)) + "/outputs/cnn_sdm_presence_only_for_debug/"
    fold = '2024-11-04_15-47-11_516294'

    fold_path = fold_dir_path + fold
    chk_path = glob.glob(fold_path + '/*.ckpt')[-1]
    path_chk=Path(chk_path)
        
    path_yaml=path_chk.parent / ".hydra/config.yaml"
    with open(path_yaml, 'r') as file:
        cfg = yaml.safe_load(file)
    cfg = OmegaConf.create(cfg)
    
    repertoire_courant_original = os.getcwd()
    os.chdir(os.path.dirname(__file__))
    cfg.data.dataset_path = os.path.abspath(cfg.data.dataset_path)
    cfg.data.csv_occurence_path = os.path.abspath(cfg.data.csv_occurence_path)
    cfg.visualization.chk_path_validate_metric = os.path.abspath(cfg.visualization.chk_path_validate_metric)
    cfg.transfer_learning.chk_path = os.path.abspath(cfg.transfer_learning.chk_path)
    cfg.transfer_learning.model_tf.chk_tf_path = os.path.abspath(cfg.transfer_learning.model_tf.chk_tf_path)
    cfg.transfer_learning.data_tf.dataset_path = os.path.abspath(cfg.transfer_learning.data_tf.dataset_path)
    cfg.transfer_learning.data_tf.csv_occurence_path = os.path.abspath(cfg.transfer_learning.data_tf.csv_occurence_path)
    os.chdir(repertoire_courant_original)
    
    # Alternative 3.1 : mise en place pour l'Alternative 3.2
    if cfg.visualization.validate_metric == True : 
        path_chk=Path(cfg.visualization.chk_path_validate_metric)
        path_yaml=path_chk.parent / "hparams.yaml"
        import yaml
        with open(path_yaml, 'r') as file:
            cfg = yaml.safe_load(file)

        cfg=OmegaConf.create(cfg)
        cfg.visualization.validate_metric = True
        cfg.visualization.chk_path_validate_metric = path_chk
        
    cls_num_list_train, patch_data_ext, cfg, cfg_modif, patch_band = Init_of_secondary_parameters(cfg=cfg)
    
    cfg.data.csv_col_class_id = 'SurveyID' #ne pas modifier

    datamodule = MicroGeoLifeCLEF2022DataModule(**cfg.data,
                                                patch_data_ext = patch_data_ext,
                                                patch_data=cfg.patch.patch_data, 
                                                patch_band_mean = cfg.patch.patch_band_mean,
                                                patch_band_sd = cfg.patch.patch_band_sd,
                                                train_augmentation = cfg.train_augmentation,
                                                test_augmentation = cfg.test_augmentation,
                                                dataloader = cfg.dataloader, )
        
    # Alternative 1 : vérification du dataloader puis STOP
    if cfg.visualization.check_dataloader == True :   
        from modules.check_dataloader import Check_dataloader
        Check_dataloader(datamodule, cfg, patch_data_ext, patch_band)
        sys.exit()
    
    cfg_model = hydra.utils.instantiate(cfg_modif.model)
    
    model = ClassificationSystem(cfg_model, **cfg.optimizer.SGD, **cfg.optimizer.scheduler, **cfg.optimizer.loss, dropout_proba = cfg.dropout_proba, cls_num_list_train=cls_num_list_train)
    #model.model.fc = torch.nn.Sequential(torch.nn.Dropout(cfg.dropout_proba), model.model.fc)
    
    # Alternative 2 : recherche du lr optimal puis STOP
    if cfg.visualization.auto_lr_finder == True :
        Auto_lr_find(model, datamodule, cfg.trainer.accelerator, cfg.trainer.devices)
        sys.exit()
    
    # entrainement 
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

    # fait un transfert leraning si activé
    if cfg.transfer_learning.transfer_learning_activated == True :
        model, datamodule, trainer = Transfer_learning_ia_biodiv(model, cfg, cfg_model, cls_num_list_train, patch_data_ext, logger)
        if cfg.transfer_learning.model_tf.load_checkpoint_for_model_tf == True and cfg.visualization.validate_metric != True :
        #if cfg.transfer_learning.model_tf.load_checkpoint_for_model_tf == True :
            chk_path = cfg.transfer_learning.model_tf.chk_tf_path
            checkpoint = torch.load(chk_path) 
            model.load_state_dict(checkpoint['state_dict'])

    # Alternative 3.2 : permet de voir les métriques assossié à un checkpoint puis STOP
    if cfg.visualization.validate_metric == True :                       
        chk_path = cfg.visualization.chk_path_validate_metric
        checkpoint = torch.load(chk_path) 
        model.load_state_dict(checkpoint['state_dict'])
        datamodule.setup(stage="fit")
        
        trainer.validate(model, datamodule=datamodule)
        trainer.test(model, datamodule=datamodule)

        shutil.rmtree(os.getcwd())
        sys.exit()   
        
    # lance l'entrainement

        

    checkpoint = torch.load(chk_path) 
    model.load_state_dict(checkpoint['state_dict'])
    datamodule.setup(stage="fit")

    data_loader = datamodule.val_dataloader()
    y_pred = []
    y_true = []  
    for inputs, labels in data_loader:
        output = model(inputs) # Feed Network
        y_pred.extend(output.tolist()) # Save Prediction
        y_true.extend(labels) 

    y_true_val_list = []
    for tensor in y_true:
        y_true_val_list.append(tensor.item())

    data_loader = datamodule.train_dataloader()
    for inputs, labels in data_loader:
        output = model(inputs) # Feed Network
        y_pred.extend(output.tolist()) # Save Prediction
        y_true.extend(labels)

    y_true_list = []
    for tensor in y_true:
        y_true_list.append(tensor.item())
   
    df = pd.DataFrame(y_pred)
    
    df.insert(0, 'subset', 'train')
    df.insert(0, 'SurveyID', y_true_list)
    df.subset[df.SurveyID.isin(y_true_val_list)]="val"
    
    if not os.path.exists(os.getcwd()[:-len(Path(os.getcwd()).name)] + 'tranfer_learning_rf/'):
        os.makedirs(os.getcwd()[:-len(Path(os.getcwd()).name)] + 'tranfer_learning_rf/')

    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    df.to_csv(os.getcwd()[:-len(Path(os.getcwd()).name)] + 'tranfer_learning_rf/' + now + '.csv', index=False)
    shutil.rmtree(os.getcwd())

if __name__ == "__main__":
    main()
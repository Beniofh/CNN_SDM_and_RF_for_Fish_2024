import os

import torch
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor

from modules.custom_metrics import MetricChallangeIABiodiv
from modules.datamodule import MicroGeoLifeCLEF2022DataModule, ClassificationSystem

from malpolon.logging import Summary


'''
notes :    
    nombre de couche -> len(list(model_chk_tf.model.named_children()))
    détail de la couche 0 ->list(model_chk_tf.model.named_children())[0]
    récuperer le nom de tout les layers
       for name, module in model_chk_tf.model.named_modules():
       if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
           print(name)
'''

def Transfer_learning_ia_biodiv(model, cfg, cfg_model, cls_num_list_train, patch_data_ext, logger):
       
    ### adaptation des paramettres d'optimisation du model pour le transfer learning
    cfg.dropout_proba = cfg.transfer_learning.dropout_proba
    cfg.optimizer.loss.loss_type = cfg.transfer_learning.optimizer_tf.loss_type
    cfg.optimizer.SGD.lr = cfg.transfer_learning.optimizer_tf.lr
    cfg.optimizer.scheduler.metric_to_track = cfg.transfer_learning.optimizer_tf.scheduler.metric_to_track
    cfg.optimizer.scheduler.mode = cfg.transfer_learning.optimizer_tf.scheduler.mode
    cfg.optimizer.scheduler.factor = cfg.transfer_learning.optimizer_tf.scheduler.factor
    cfg.optimizer.scheduler.patience = cfg.transfer_learning.optimizer_tf.scheduler.patience
    cfg.optimizer.scheduler.threshold = cfg.transfer_learning.optimizer_tf.scheduler.threshold
    cfg.optimizer.scheduler.cooldown = cfg.transfer_learning.optimizer_tf.scheduler.cooldown
    cfg.optimizer.scheduler.logging_interval = cfg.transfer_learning.optimizer_tf.scheduler.logging_interval
                
    ### chargement des poids et aplication de la modification des parametres d'optimisation du model pour le transfer learning
    if cfg.transfer_learning.load_checkpoint == True :
        model_chk_tf = model.load_from_checkpoint(cfg.transfer_learning.chk_path, model=cfg_model, **cfg.optimizer.SGD, **cfg.optimizer.scheduler, **cfg.optimizer.loss, dropout_proba = cfg.dropout_proba, cls_num_list_train=cls_num_list_train)
    else :
        model_chk_tf = ClassificationSystem(cfg_model, **cfg.optimizer.SGD, **cfg.optimizer.scheduler, **cfg.optimizer.loss, dropout_proba = cfg.dropout_proba, cls_num_list_train=cls_num_list_train)
      
    ### geler touts layers
    if cfg.transfer_learning.model_tf.fize_all_layer == True :
        for param in model_chk_tf.model.parameters():
            param.requiers_grad = False

    ### dégeler des layers
    for num_layer in cfg.transfer_learning.model_tf.unfreeze_layer :
        eval('model_chk_tf.model.layer' + str(num_layer) + '.requires_grad_(True)') 
    
    ### récupération du nombre de features du model d'origine
    num_ftrs_in = model_chk_tf.model.fc[1].in_features
    
    ### remplacement de la dernière couche et modification du nombre de sortie  
    model_chk_tf.model.fc = torch.nn.Sequential(model_chk_tf.model.fc[0], 
                                                torch.nn.Linear(num_ftrs_in, cfg.transfer_learning.model_tf.num_outputs_tf),
                                                torch.nn.ReLU())
    '''

    model_chk_tf.model.fc = torch.nn.Sequential(model_chk_tf.model.fc[0], 
                                                torch.nn.Linear(num_ftrs_in, cfg.transfer_learning.model_tf.num_outputs_tf))
    '''
                                            
    ### définition des nouvelles métriques
    model_chk_tf.metrics = {"r2score_matrix_log": "",
                            "mean_absolute_error": torchmetrics.MeanAbsoluteError().to(device = "cuda"),
                            "mean_absolute_log_error": torchmetrics.MeanAbsoluteError().to(device = "cuda"),
                            "mean_squared_error" : torchmetrics.MeanSquaredError().to(device = "cuda"),
                            "mean_squared_log_error" : torchmetrics.MeanSquaredLogError().to(device = "cuda"),
                            "r2score_mean_by_site" : "",
                            "r2score_mean_by_site_of_log" : ""}
    ### /!\ note importante pour "r2score_mean_by_site" et "r2score_mean_by_site_of_log"
    #-> ici, seul le nom "r2score_mean_by_site" est juste rajouter au dictionaire model_chk_tf.metrics
    #-> la définition de la métrique en soit ce fait dans standard_prediction_systems.py au niveau de 
    # "if metric_name == "r2score_mean_by_site""  
    #-> cela est obligatoire car pour faire cette metrique il faud transposer y et y_hat et donc le 
    # num_outputs de torchmetrics.R2Score va valoir y_trans.shape[1]. Or cela correspond au nombre de 
    # site n'est pas le même entre le test et le train set d'où l'obligation de définir num_outputs dans
    # standard_prediction_systems.py

    ### adaptation des paramettres pour le chargement du nouveau jeu de donnée
    data_tf = cfg.data.copy()
    data_tf.dataset_path = cfg.transfer_learning.data_tf.dataset_path
    data_tf.csv_occurence_path = cfg.transfer_learning.data_tf.csv_occurence_path
    data_tf.csv_separator = cfg.transfer_learning.data_tf.csv_separator
    data_tf.csv_col_occurence_id = cfg.transfer_learning.data_tf.csv_col_occurence_id
    data_tf.csv_col_class_id = cfg.transfer_learning.data_tf.csv_col_class_id
    data_tf.train_batch_size = 32
    data_tf.inference_batch_size = 256
    data_tf.num_workers = 8

    ### configuration d'un datamodule pour le nouveau jeu de donnée sur la base de l'adaptation des paramettres pour le chargement du nouveau jeu de donnée 
    datamodule_tf = MicroGeoLifeCLEF2022DataModule(**data_tf,
                                                   patch_data_ext = patch_data_ext,
                                                   patch_data=cfg.patch.patch_data, 
                                                   patch_band_mean = cfg.patch.patch_band_mean,
                                                   patch_band_sd = cfg.patch.patch_band_sd,
                                                   train_augmentation = cfg.train_augmentation,
                                                   test_augmentation = cfg.test_augmentation,
                                                   dataloader = cfg.dataloader,)
            
    ### adaptation des paramettres pour les callbacks
    cfg.callbacks.monitor = cfg.transfer_learning.callbacks_tf.monitor
    cfg.callbacks.mode = cfg.transfer_learning.callbacks_tf.mode
    cfg.callbacks.patience = cfg.transfer_learning.callbacks_tf.patience

    ### configuration des callbacks pour appliquer les adaptation des paramettres pour les callbacks
    callbacks_tf = [
        Summary(),
        ModelCheckpoint(
            dirpath=os.getcwd(),
            filename="checkpoint-{epoch:02d}-{step}-{val_r2score_mean_by_site_of_log:.4f}",
            monitor= cfg.callbacks.monitor,                    
            mode=cfg.callbacks.mode,),
        LearningRateMonitor(logging_interval=cfg.optimizer.scheduler.logging_interval), #epoch'),
        EarlyStopping(monitor=cfg.callbacks.monitor, min_delta=0.00, patience=cfg.callbacks.patience, verbose=False, mode=cfg.callbacks.mode),]              
          
    ### configuration du trainer pour appliquer la configuration des callbacks
    trainer_tf = pl.Trainer(logger=logger, callbacks=callbacks_tf, **cfg.trainer)
    return model_chk_tf, datamodule_tf, trainer_tf
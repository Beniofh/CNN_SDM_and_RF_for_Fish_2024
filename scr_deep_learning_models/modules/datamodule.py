import sys
import torch
import torchmetrics

from torchvision import transforms
from typing import Mapping, Union

from malpolon.data.data_module import BaseDataModule
from malpolon.models.standard_prediction_systems import GenericPredictionSystemLrScheduler
from malpolon.models.utils import check_model

from modules.dataset import MicroGeoLifeCLEF2022Dataset
from modules.transforms import *
from pytopk import BalNoisedTopK
from pytopk import ImbalNoisedTopK
from modules.koleo_loss import KoLeoLoss

from modules.custom_metrics import MacroAverageTopK_Maximilien


class PreprocessData():
    def __init__(self, patch_band_mean, patch_band_sd):
        self.patch_band_mean = patch_band_mean
        self.patch_band_sd = patch_band_sd

    def __call__(self, data ):       
        
        # mpa
        if "mpa" in list(data.keys()) : 
            mpa_data = data["mpa"]
            mpa_data = Only_Tensor_Transform()(mpa_data)
            mpa_data = transforms.Normalize(self.patch_band_mean["mpa"], self.patch_band_sd["mpa"])(mpa_data)

        # fishing_pressure
        if "fishing_pressure" in list(data.keys()) : 
            fishing_pressure_data = data["fishing_pressure"]
            fishing_pressure_data = Only_Tensor_Transform()(fishing_pressure_data)

        # bathymetry
        if "bathymetry" in list(data.keys()) : 
            bathymetry_data = data["bathymetry"]
            bathymetry_data = Bathymetry_Transform()(bathymetry_data)
            bathymetry_data = transforms.Normalize(self.patch_band_mean["bathymetry"], self.patch_band_sd["bathymetry"])(bathymetry_data)
        
        # bathy_95m
        if "bathy_95m" in list(data.keys()) : 
            bathy_95m_data = data["bathy_95m"]
            bathy_95m_data = Baty_95m_Transform()(bathy_95m_data)
            bathy_95m_data = transforms.Normalize(self.patch_band_mean["bathy_95m"], self.patch_band_sd["bathy_95m"])(bathy_95m_data)

        # chlorophyll_concentration_1km
        if "chlorophyll_concentration_1km" in list(data.keys()) : 
            chlorophyll_concentration_1km_data = data["chlorophyll_concentration_1km"]
            chlorophyll_concentration_1km_data = Chlorophyll_Concentration_1km_Transform()(chlorophyll_concentration_1km_data)
            chlorophyll_concentration_1km_data = transforms.Normalize(self.patch_band_mean["chlorophyll_concentration_1km"], self.patch_band_sd["chlorophyll_concentration_1km"])(chlorophyll_concentration_1km_data)

        # full_true_clean_subset
        if "full_true_clean_subset" in list(data.keys()) : 
            full_true_clean_subset_data = data["full_true_clean_subset"]
            full_true_clean_subset_data = Only_Tensor_Transform()(full_true_clean_subset_data)

        # meditereanean_sst
        if "meditereanean_sst" in list(data.keys()) : 
            meditereanean_sst_data = data["meditereanean_sst"]
            meditereanean_sst_data = Meditereanean_Sst_Transform()(meditereanean_sst_data)
            meditereanean_sst_data = transforms.Normalize(self.patch_band_mean["meditereanean_sst"], self.patch_band_sd["meditereanean_sst"])(meditereanean_sst_data)

        # north_water_velocity_4_2km_mean and east_water_velocity_4_2km_mean
        standard_3_bands_15_pixels_elements = [element for element in list(data.keys()) if "_water_velocity_4_2km_mean" in element]
        if standard_3_bands_15_pixels_elements!=0 :
            for patch_standard_3_bands_15_pixels_elements in standard_3_bands_15_pixels_elements:
                patch_standard_3_bands_15_pixels_elements_data = data[patch_standard_3_bands_15_pixels_elements]
                patch_standard_3_bands_15_pixels_elements_data = Standard_3_Bands_15_Pixels_Transform()(patch_standard_3_bands_15_pixels_elements_data)
                patch_standard_3_bands_15_pixels_elements_data = transforms.Normalize(self.patch_band_mean[patch_standard_3_bands_15_pixels_elements], self.patch_band_sd[patch_standard_3_bands_15_pixels_elements])(patch_standard_3_bands_15_pixels_elements_data)
                exec(patch_standard_3_bands_15_pixels_elements+'_data = torch.clone(patch_standard_3_bands_15_pixels_elements_data)')
        
        # occ_lat_long
        if "occ_lat_long" in list(data.keys()) : 
            occ_lat_long_data = data["occ_lat_long"]
            occ_lat_long_data = Only_Tensor_Transform()(occ_lat_long_data)
            #occ_lat_long_data = transforms.Normalize(self.patch_band_mean["occ_lat_long"], self.patch_band_sd["occ_lat_long"])(occ_lat_long_data)

        # salinity_4_2km_mean
        standard_3_bands_30_to_15_pixels_elements = [element for element in list(data.keys()) if "salinity_4_2km_mean" in element]
        if standard_3_bands_30_to_15_pixels_elements!=0 :
            for patch_standard_3_bands_30_to_15_pixels_elements in standard_3_bands_30_to_15_pixels_elements:
                patch_standard_3_bands_30_to_15_pixels_elements_data = data[patch_standard_3_bands_30_to_15_pixels_elements]
                patch_standard_3_bands_30_to_15_pixels_elements_data = Standard_3_Bands_30_To_15_Pixels_Transform()(patch_standard_3_bands_30_to_15_pixels_elements_data)
                patch_standard_3_bands_30_to_15_pixels_elements_data = transforms.Normalize(self.patch_band_mean[patch_standard_3_bands_30_to_15_pixels_elements], self.patch_band_sd[patch_standard_3_bands_30_to_15_pixels_elements])(patch_standard_3_bands_30_to_15_pixels_elements_data)
                exec(patch_standard_3_bands_30_to_15_pixels_elements+'_data = torch.clone(patch_standard_3_bands_30_to_15_pixels_elements_data)')
        
        # Sea_water_potential_temperature_at_sea_floor_4_2km_mean
        standard_1_bands_15_pixels_elements = [element for element in list(data.keys()) if "sea_water_potential_temperature_at_sea_floor_4_2km_mean" in element]
        if standard_1_bands_15_pixels_elements!=0 :
            for patch_standard_1_bands_15_pixels_elements in standard_1_bands_15_pixels_elements:
                patch_standard_1_bands_15_pixels_elements_data = data[patch_standard_1_bands_15_pixels_elements]
                patch_standard_1_bands_15_pixels_elements_data = Standard_1_Bands_15_Pixels_Transform()(patch_standard_1_bands_15_pixels_elements_data)
                patch_standard_1_bands_15_pixels_elements_data = transforms.Normalize(self.patch_band_mean[patch_standard_1_bands_15_pixels_elements], self.patch_band_sd[patch_standard_1_bands_15_pixels_elements])(patch_standard_1_bands_15_pixels_elements_data)
                exec(patch_standard_1_bands_15_pixels_elements+'_data = torch.clone(patch_standard_1_bands_15_pixels_elements_data)')

        # substrate
        if "substrate" in list(data.keys()) : 
            substrate_data = data["substrate"]
            substrate_data = Substrate_Transform()(substrate_data)

        # TCI_sentinel et B01_sentinel à B12_sentinel
        sentinel_elements = [element for element in list(data.keys()) if "sentinel" in element]
        if sentinel_elements!=0 :
            for patch_sentinel in sentinel_elements:
                patch_sentinel_data = data[patch_sentinel]
                patch_sentinel_data = Sentinel_Transform()(patch_sentinel_data)
                patch_sentinel_data = transforms.Normalize(self.patch_band_mean[patch_sentinel], self.patch_band_sd[patch_sentinel])(patch_sentinel_data)                
                exec(patch_sentinel+'_data = torch.clone(patch_sentinel_data)')

        str_patch_data = ""
        for patch_data in list(data.keys()):
            str_patch_data += patch_data + "_data, "

        concat_data = eval("torch.concat((" + str_patch_data + "))")
        #concat_data = transforms.RandomRotation(degrees=(24, 25))(concat_data)
        #concat_data = transforms.RandomCrop(size=(212, 212))(concat_data)
        #concat_data = transforms.Resize(256)(concat_data)
        
        return concat_data

class MicroGeoLifeCLEF2022DataModule(BaseDataModule):
    """
    Data module for MicroGeoLifeCLEF 2022.

    Parameters
    ----------
        dataset_path: Path to dataset
        train_batch_size: Size of batch for training
        inference_batch_size: Size of batch for inference (validation, testing, prediction)
        num_workers: Number of workers to use for data loading
    """
    def __init__(
        self,
        dataset_path: str,
        csv_occurence_path: str,
        csv_separator:str,
        csv_col_occurence_id:str,
        csv_col_class_id: str,
        train_batch_size: int,
        inference_batch_size: int,
        num_workers: int,
        patch_data_ext: list,
        patch_data: list,
        patch_band_mean: dict,
        patch_band_sd: dict,
        train_augmentation: dict,
        test_augmentation: dict,
        dataloader: dict,
    ):
        super().__init__(train_batch_size, inference_batch_size, num_workers, dataloader)
        self.dataset_path = dataset_path
        self.csv_occurence_path = csv_occurence_path
        self.csv_separator = csv_separator
        self.csv_col_class_id = csv_col_class_id
        self.csv_col_occurence_id = csv_col_occurence_id
        self.patch_data_ext = patch_data_ext
        self.patch_data = patch_data
        self.patch_band_mean = patch_band_mean
        self.patch_band_sd = patch_band_sd
        self.train_augmentation = train_augmentation
        self.test_augmentation = test_augmentation

    @property
    def train_transform(self):
        transform_base = transforms.Compose([PreprocessData(self.patch_band_mean, self.patch_band_sd),])
        
        if self.train_augmentation.random_rotation.tr_rr_activation == True : 
            transform_rr = transforms.Compose([transforms.RandomRotation(degrees=eval(self.train_augmentation.random_rotation.tr_rr_degrees)),
                                               transforms.CenterCrop(self.train_augmentation.random_rotation.tr_rr_center_crop)])
        else : 
            transform_rr = transforms.Compose([])
        
        if self.train_augmentation.crop.tr_c_activation == 'random_crop' :         
            transform_c = transforms.Compose([transforms.RandomCrop(size=eval(self.train_augmentation.crop.tr_c_size)),])
        elif self.train_augmentation.crop.tr_c_activation == 'center_crop' :
            transform_c = transforms.Compose([transforms.CenterCrop(size=eval(self.train_augmentation.crop.tr_c_size)),])           
        else : 
            transform_c = transforms.Compose([])

        if self.train_augmentation.resize.tr_r_activation == True : 
            transform_resize = transforms.Compose([transforms.Resize(self.train_augmentation.resize.tr_r_size),])
        else : 
            transform_resize = transforms.Compose([])

        transform_final = transforms.Compose([transform_base, transform_rr, transform_c, transform_resize])
        
        return transform_final

    # test_transform est appellé pour le test ET LE VAL    
    @property
    def test_transform(self):
        transform_base = transforms.Compose([PreprocessData(self.patch_band_mean, self.patch_band_sd),])
        
        if self.test_augmentation.random_rotation.te_rr_activation == True : 
            transform_rr = transforms.Compose([transforms.RandomRotation(degrees=eval(self.test_augmentation.random_rotation.te_rr_degrees)),
                                               transforms.CenterCrop(self.test_augmentation.random_rotation.te_rr_center_crop)])
        else : 
            transform_rr = transforms.Compose([])
        
        if self.test_augmentation.crop.te_c_activation == 'random_crop' :         
            transform_c = transforms.Compose([transforms.RandomCrop(size=eval(self.test_augmentation.crop.te_c_size)),])
        elif self.test_augmentation.crop.te_c_activation == 'center_crop' :
            transform_c = transforms.Compose([transforms.CenterCrop(size=eval(self.test_augmentation.crop.te_c_size)),])           
        else : 
            transform_c = transforms.Compose([])

        if self.test_augmentation.resize.te_r_activation == True : 
            transform_resize = transforms.Compose([transforms.Resize(self.test_augmentation.resize.te_r_size),])
        else : 
            transform_resize = transforms.Compose([])

        transform_final = transforms.Compose([transform_base, transform_rr, transform_c, transform_resize])
        
        return transform_final
    
    '''
    @property
    def train_transform(self):       
        return transforms.Compose(
            [
                PreprocessData(self.patch_band_mean, self.patch_band_sd),
            ]
        )

    # test_transform est appellé pour le test ET LE VAL    
    @property
    def test_transform(self):
        return transforms.Compose(
            [
                PreprocessData(self.patch_band_mean, self.patch_band_sd),
            ]
        )
    '''

    def prepare_data(self):
        MicroGeoLifeCLEF2022Dataset(
            self.dataset_path,
            csv_separator = self.csv_separator,
            csv_col_class_id = self.csv_col_class_id,
            csv_col_occurence_id = self.csv_col_occurence_id,
            patch_data_ext = self.patch_data_ext,
            subset = "train",
            use_rasters=False,
            csv_occurence_path = self.csv_occurence_path 
        )

    def get_dataset(self, split, transform, **kwargs):
        dataset = MicroGeoLifeCLEF2022Dataset(
            self.dataset_path,
            csv_separator = self.csv_separator,
            csv_col_class_id = self.csv_col_class_id,
            csv_col_occurence_id = self.csv_col_occurence_id,
            patch_data_ext = self.patch_data_ext,
            subset = split,
            patch_data=self.patch_data,
            use_rasters=False,
            csv_occurence_path = self.csv_occurence_path,
            transform=transform,
            **kwargs
        )
        return dataset

class ClassificationSystem(GenericPredictionSystemLrScheduler):
    def __init__(
        self,
        model: Union[torch.nn.Module, Mapping],
        lr: float,
        weight_decay: float,
        momentum: float,
        nesterov: bool,
        mode: str,
        factor: float,
        patience: int,
        threshold: float, 
        cooldown: int,
        logging_interval: str,
        metric_to_track: str, 
        loss_type: str,
        k: int,
        epsilon: float,
        max_m: float,
        dropout_proba: float,
        cls_num_list_train: list):

        num_outputs = model.modifiers.change_last_layer.num_outputs
        model = check_model(model)
        
        if loss_type == 'BalNoisedTopK':
            loss = BalNoisedTopK(k=k, epsilon=epsilon)
        elif loss_type == 'ImbalNoisedTopK':
            from init_elements import NormedLinear
            model.fc = NormedLinear(model.fc.in_features, model.fc.out_features)
            loss =  ImbalNoisedTopK(k=k, epsilon=epsilon, max_m=max_m, cls_num_list=cls_num_list_train)
        elif loss_type == 'PoissonNLLLoss':
            loss = torch.nn.PoissonNLLLoss(log_input=True, full=True)
        elif loss_type == 'L1Loss':
            loss = torch.nn.L1Loss()
        elif loss_type == 'L2Loss':
            loss = torch.nn.MSELoss()
        elif loss_type == 'KoLeoLoss':               
            loss = [KoLeoLoss(), torch.nn.CrossEntropyLoss()]
        elif loss_type == 'CrossEntropyLoss':
            loss = torch.nn.CrossEntropyLoss()
        else :
            print("La loss '" + loss_type +"' n'est pas géré." ) 
            sys.exit(1)

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,)
        
        scheduler = {'lr_scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience, cooldown=cooldown, threshold=threshold),
                     'metric_to_track': metric_to_track}
        
        # Ajout dropout
        model.fc = torch.nn.Sequential(torch.nn.Dropout(dropout_proba), model.fc)
        
        '''
        # test pour mise en place d'un scheduler de type CyclicLR
        # marche bien mais s'update uniquement sur les époques et non les steps

        scheduler = {'lr_scheduler': torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.00001, max_lr=0.01, step_size_up=4, step_size_down=None, mode="exp_range",gamma=0.95)}
        scheduler['metric_to_track']= metric_to_track
        # instantiate the WeakMethod in the lr scheduler object into the custom scale function attribute
        scheduler['lr_scheduler']._scale_fn_custom = scheduler['lr_scheduler']._scale_fn_ref()
        # remove the reference so there are no more WeakMethod references in the object
        scheduler['lr_scheduler']._scale_fn_ref = None
        '''

        metrics = {"accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=num_outputs, top_k=1).to(device = "cuda")}            
        #metrics['accuracy_macro'] = torchmetrics.Accuracy(task="multiclass",num_classes=num_outputs, average="macro", top_k=1).to(device = "cuda")
        metrics['accuracy_macro'] = MacroAverageTopK_Maximilien(k=1).to(device = "cuda")

        if num_outputs > 5 :
            metrics["top_5_accuracy"] = torchmetrics.Accuracy(task="multiclass", num_classes=num_outputs, top_k=5).to(device = "cuda")
            #metrics["top_5_accuracy_macro"] = torchmetrics.Accuracy(task="multiclass",num_classes=num_outputs, average="macro", top_k=5).to(device = "cuda")
            metrics['top_5_accuracy_macro'] = MacroAverageTopK_Maximilien(k=5).to(device = "cuda")

        if num_outputs > 10 :
            metrics["top_10_accuracy"] = torchmetrics.Accuracy(task="multiclass", num_classes=num_outputs, top_k=10).to(device = "cuda")
            #metrics["top_10_accuracy_macro"] = torchmetrics.Accuracy(task="multiclass",num_classes=num_outputs, average="macro", top_k=10).to(device = "cuda")
            metrics['top_10_accuracy_macro'] = MacroAverageTopK_Maximilien(k=10).to(device = "cuda")

        if num_outputs > 20 :            
            metrics["top_20_accuracy"] = torchmetrics.Accuracy(task="multiclass", num_classes=num_outputs, top_k=20).to(device = "cuda")
            #metrics["top_20_accuracy_macro"] = torchmetrics.Accuracy(task="multiclass",num_classes=num_outputs, average="macro", top_k=20).to(device = "cuda")
            metrics['top_20_accuracy_macro'] = MacroAverageTopK_Maximilien(k=20).to(device = "cuda")


        
        super().__init__(model, loss, optimizer, scheduler, metrics)

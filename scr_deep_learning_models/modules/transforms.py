import numpy as np
from torchvision import transforms
import torch

    
class Baty_95m_Transform :                                                  
    def __call__(self, data):
        #data = np.tile(data[:, :, None], 3)
        data = transforms.functional.to_tensor(data)
        if np.nanmean(np.isnan(transforms.CenterCrop(31)(data))) < 1 : 
            mean_data = np.nanmean(transforms.CenterCrop(31)(data))
        elif np.nanmean(np.isnan(transforms.CenterCrop(41)(data))) < 1 : 
            mean_data = np.nanmean(transforms.CenterCrop(41)(data))
        elif np.nanmean(np.isnan(transforms.CenterCrop(51)(data))) < 1 : 
            mean_data = np.nanmean(transforms.CenterCrop(51)(data))
        elif np.nanmean(np.isnan(transforms.CenterCrop(61)(data))) < 1 : 
            mean_data = np.nanmean(transforms.CenterCrop(61)(data))            
        elif np.nanmean(np.isnan(transforms.CenterCrop(71)(data))) < 1 : 
            mean_data = np.nanmean(transforms.CenterCrop(71)(data)) 
        else :
            mean_data = np.nanmean(transforms.CenterCrop(80)(data))         
        np.nan_to_num(data, copy=False, nan=np.nanmean(mean_data))   
        data = transforms.CenterCrop(31)(data)
        data = transforms.Resize(256, interpolation=transforms.InterpolationMode.NEAREST)(data)
        return data

class Bathymetry_Transform :                                                  
    def __call__(self, data):
        data = transforms.functional.to_tensor(data)  
        data = transforms.CenterCrop(20)(data)
        data = transforms.Resize(256, interpolation=transforms.InterpolationMode.NEAREST)(data)
        return data

class Chlorophyll_Concentration_1km_Transform :                                                  
    def __call__(self, data):
        #data = np.tile(data[:, :, None], 3)
        data = transforms.functional.to_tensor(data)
        if np.nanmean(np.isnan(transforms.CenterCrop(32)(data))) < 1 : 
            mean_data = np.nanmean(transforms.CenterCrop(32)(data))
        elif np.nanmean(np.isnan(transforms.CenterCrop(42)(data))) < 1 : 
            mean_data = np.nanmean(transforms.CenterCrop(42)(data))
        elif np.nanmean(np.isnan(transforms.CenterCrop(52)(data))) < 1 : 
            mean_data = np.nanmean(transforms.CenterCrop(52)(data))
        elif np.nanmean(np.isnan(transforms.CenterCrop(62)(data))) < 1 : 
            mean_data = np.nanmean(transforms.CenterCrop(62)(data))
        else : 
            mean_data = np.nanmean(transforms.CenterCrop(72)(data))     
        np.nan_to_num(data, copy=False, nan=np.nanmean(mean_data))   
        data = transforms.CenterCrop(30)(data)
        data = transforms.Resize(256, interpolation=transforms.InterpolationMode.NEAREST)(data)
        return data

class Meditereanean_Sst_Transform :                                                  
    def __call__(self, data):
        #data = np.tile(data[:, :, None], 3)
        data = transforms.functional.to_tensor(data)
        if np.nanmean(np.isnan(transforms.CenterCrop(32)(data))) < 1 : 
            mean_data = np.nanmean(transforms.CenterCrop(32)(data))
        elif np.nanmean(np.isnan(transforms.CenterCrop(42)(data))) < 1 : 
            mean_data = np.nanmean(transforms.CenterCrop(42)(data))
        elif np.nanmean(np.isnan(transforms.CenterCrop(52)(data))) < 1 : 
            mean_data = np.nanmean(transforms.CenterCrop(52)(data))
        elif np.nanmean(np.isnan(transforms.CenterCrop(62)(data))) < 1 : 
            mean_data = np.nanmean(transforms.CenterCrop(62)(data))
        else :
            mean_data = np.nanmean(transforms.CenterCrop(65)(data))
        np.nan_to_num(data, copy=False, nan=np.nanmean(mean_data))   
        data = transforms.CenterCrop(32)(data)
        data = transforms.Resize(256, interpolation=transforms.InterpolationMode.NEAREST)(data)
        return data
    
class Occ_Lat_Long_Transform:
    def __call__(self, data):
        data = transforms.functional.to_tensor(data)
        data = transforms.Resize(256, interpolation=transforms.InterpolationMode.NEAREST)(data)
        return data

class Only_Tensor_Transform :                                                  
    def __call__(self, data):
        data = transforms.functional.to_tensor(data)
        return data
    
class Sentinel_Transform:
    def __call__(self, data):
        data = transforms.functional.to_tensor(data) 
        data = transforms.CenterCrop(size=300)(data)
        data = transforms.Resize(256)(data)
        return data 

class Standard_1_Bands_15_Pixels_Transform :                                                  
    def __call__(self, data):
        data = transforms.functional.to_tensor(data)
        data = transforms.CenterCrop(15)(data)
        np.nan_to_num(data, copy=False, nan=np.nanmean(data))      
        data = transforms.Resize(256, interpolation=transforms.InterpolationMode.NEAREST)(data)
        return data
    
'''
Note pour class Standard_3_Bands_15_Pixels_Transform : 
    Le problème de NaN indiqué dans le print est très minoritaire (conserne 2-3 occurences).
    Dans l'idéal, il faidrait télécharger des patches de 30 et demander la moyer sur les 30 quand pas dispo sur 15.
'''
class Standard_3_Bands_15_Pixels_Transform :                                                  
    def __call__(self, data):
        data = transforms.functional.to_tensor(data)
        
        mean_data_0 = np.nanmean(transforms.CenterCrop(15)(data[0,:,:]))
        if np.nanmean(np.isnan(transforms.CenterCrop(15)(data[1,:,:]))) < 1 :
            mean_data_1 = np.nanmean(transforms.CenterCrop(15)(data[1,:,:]))
        else :
            mean_data_1 = mean_data_0
            print('Warning: for a raster of one occurrence, it was detected during the "Standard_3_Bands_15_Pixels_Transform" that all values of transforms.CenterCrop(15)(data[1, :,:]) are NaN. The NaN of transforms.CenterCrop(15)(data[1, :,:]) have been replaced by np.nanmean(transforms.CenterCrop(15)(data[0, :,:])')
        if np.nanmean(np.isnan(transforms.CenterCrop(15)(data[2,:,:]))) < 1 :
            mean_data_2 = np.nanmean(transforms.CenterCrop(15)(data[2,:,:]))
        else :
            mean_data_2 = mean_data_1
            print('Warning: for a raster of one occurrence, it was detected during the "Standard_3_Bands_15_Pixels_Transform" that all values of transforms.CenterCrop(15)(data[1, :,:]) are NaN. The NaN of transforms.CenterCrop(15)(data[1, :,:]) have been replaced by np.nanmean(transforms.CenterCrop(15)(data[0, :,:])')

        np.nan_to_num(data[0,:,], copy=False, nan=np.nanmean(mean_data_0))
        np.nan_to_num(data[1,:,], copy=False, nan=np.nanmean(mean_data_1))   
        np.nan_to_num(data[2,:,], copy=False, nan=np.nanmean(mean_data_2))      
        data = transforms.CenterCrop(15)(data)
        data = transforms.Resize(256, interpolation=transforms.InterpolationMode.NEAREST)(data)
        return data
    
class Standard_3_Bands_30_To_15_Pixels_Transform :                                                  
    def __call__(self, data):
        data = transforms.functional.to_tensor(data)
        
        if np.nanmean(np.isnan(transforms.CenterCrop(15)(data[0,:,:]))) < 1 :
            mean_data_0 = np.nanmean(transforms.CenterCrop(15)(data[0,:,:]))
        elif np.nanmean(np.isnan(transforms.CenterCrop(20)(data[0,:,:]))) < 1 :
            mean_data_0 = np.nanmean(transforms.CenterCrop(20)(data[0:,:,]))
        else :
            mean_data_0 = np.nanmean(transforms.CenterCrop(30)(data[0:,:,]))
        
        if np.nanmean(np.isnan(transforms.CenterCrop(15)(data[1,:,:]))) < 1 :
            mean_data_1 = np.nanmean(transforms.CenterCrop(15)(data[1,:,:]))
        elif np.nanmean(np.isnan(transforms.CenterCrop(20)(data[1,:,:]))) < 1 :
            mean_data_1 = np.nanmean(transforms.CenterCrop(20)(data[1:,:,]))
        else :
            mean_data_1 = np.nanmean(transforms.CenterCrop(30)(data[1:,:,]))

        if np.nanmean(np.isnan(transforms.CenterCrop(15)(data[2,:,:]))) < 1 :
            mean_data_2 = np.nanmean(transforms.CenterCrop(15)(data[2,:,:]))
        elif np.nanmean(np.isnan(transforms.CenterCrop(20)(data[2,:,:]))) < 1 :
            mean_data_2 = np.nanmean(transforms.CenterCrop(20)(data[2:,:,]))
        else :
            mean_data_2 = np.nanmean(transforms.CenterCrop(30)(data[2:,:,]))        
        
        np.nan_to_num(data[0,:,], copy=False, nan=np.nanmean(mean_data_0))
        np.nan_to_num(data[1,:,], copy=False, nan=np.nanmean(mean_data_1))   
        np.nan_to_num(data[2,:,], copy=False, nan=np.nanmean(mean_data_2))      
    
        data = transforms.CenterCrop(15)(data)
        data = transforms.Resize(256, interpolation=transforms.InterpolationMode.NEAREST)(data)
        return data
    
class Substrate_Transform :                                                  
    def __call__(self, data):
        data = np.tile(data[:, :, None], 8)
        data = transforms.functional.to_tensor(data)
        for x in range(0,8):
            if x==1 :
                data[x,:,][data[x,:,]==x] = 99
                data[x,:,][torch.logical_and(data[0,:,] != -1, data[0,:,] != 99)]=1
                data[x,:,][data[x,:,]==99] = 0
            else :
                data[x,:,][torch.logical_and(data[x,:,] != -1, data[x,:,] != x)]=1
                data[x,:,][data[x,:,]==x] = 0
        return data


        
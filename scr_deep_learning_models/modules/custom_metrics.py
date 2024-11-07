import torch
from torchmetrics import Metric


class MetricChallangeIABiodiv(Metric):
    
    def __init__(self):
        super().__init__() 
        # Initialisation des accumulateurs 
        self.species = self.add_state("sum_of_dif", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.diff = self.add_state("n_occ", default=torch.tensor(0.0), dist_reduce_fx="sum")
        
    def update(self, y_hat, y):
        y_temp=torch.clone(y)
        # revient à faire max(1-y)
        y_temp[y_temp<1]=1

        y_hat_temp=torch.clone(y_hat)
        # revient à faire max(1-ŷ)
        y_hat_temp[y_hat_temp<1]=1 
        
        # la somme de la valeur absolue de log10(ŷ)-log10(y)
        sum_of_dif=torch.sum(torch.abs(torch.log10(y_hat_temp)-torch.log10(y_temp)))        
        # le nombre d'occurence
        n_occ = y_temp.numel()
        # Mise à jour des accumulateurs
        self.sum_of_dif += sum_of_dif
        self.n_occ += n_occ
    
    def compute(self):
        # Calcul de la métrique finale

        return self.sum_of_dif / self.n_occ
    



import pandas as pd
from sklearn.metrics import top_k_accuracy_score

class MacroAverageTopK_Maximilien (Metric):
    def __init__(self, 
                 k, 
                 dist_sync_on_step=True):
        super().__init__(dist_sync_on_step=dist_sync_on_step) 
        self.k = k
        # Initialisation des accumulateurs 
        self.add_state("lables", default=torch.tensor([]), dist_reduce_fx="sum")
        self.add_state("predictions", default=torch.tensor([]), dist_reduce_fx="sum")

    def update(self, y_hat, y):
        self.lables = torch.cat((self.lables, y), dim=0)
        self.predictions = torch.cat((self.predictions, y_hat), dim=0)       
    
    def compute(self):
        
        # Calcul de la métrique finale
        dfw = pd.DataFrame(self.lables.cpu().numpy(), columns=['labels'])
        dfw['id'] = dfw.index       
        weights = dfw.groupby('labels')\
                     .count()[['id']]\
                     .apply(lambda a: 1/a).rename({'id': 'weight'}, axis=1)
        weights.columns = ['weight']
        dfw = dfw.join(weights, how='left', on='labels')
        y = dfw.labels
        y_hat = self.predictions.detach().cpu().numpy()
        macro_average_topk = top_k_accuracy_score(y,
                                    y_hat,
                                    k=self.k,
                                    labels=range(y_hat.shape[1]),
                                    sample_weight=dfw.weight)
        return macro_average_topk




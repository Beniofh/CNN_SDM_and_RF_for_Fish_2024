import os
import shutil
from pathlib import Path
from matplotlib import pyplot as plt
import datetime

import pytorch_lightning as pl    

# https://www.youtube.com/watch?v=WMp-Fu2mlj8
def Auto_lr_find(model, datamodule, accelerator, devices):
    trainer = pl.Trainer(auto_lr_find=True, accelerator=accelerator, devices=devices)
            
    lr_finder = trainer.tuner.lr_find(model, datamodule=datamodule, min_lr=1e-6)
    # Results can be found in -> lr_finder.results

    # Plot with
    fig = lr_finder.plot(suggest=True)
    plt.title('Learning rate suggestion: '+ str(lr_finder.suggestion()))

    if not os.path.exists(os.getcwd()[:-len(Path(os.getcwd()).name)] + 'auto_lr_finder/'):
        os.makedirs(os.getcwd()[:-len(Path(os.getcwd()).name)] + 'auto_lr_finder/')
    
    now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
    fig.savefig(os.getcwd()[:-len(Path(os.getcwd()).name)] + 'auto_lr_finder/' + 'auto_lr_finder_' + now + '.png' )
    
    shutil.rmtree(os.getcwd())    
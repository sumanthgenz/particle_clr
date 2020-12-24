import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from main import *

if __name__ == "__main__":

  model = SupervisedModel()
  wandb_logger.watch(model, log='gradients', log_freq=10)

  trainer = pl.Trainer(
      default_root_dir='/home/sgurram/Desktop/pcl_checkpoint', 
      gpus=2, 
      max_epochs=100, 
      logger=wandb_logger, 
      distributed_backend='ddp')  

  trainer.fit(model)

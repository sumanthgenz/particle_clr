import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from metrics import *

class NCELoss(torch.nn.Module):
    def __init__(self, temperature=0.5):
        super(NCELoss, self).__init__()
        self.temperature = 0.5
    
    def forward(self, x, y):
        loss = nce_loss

        return loss(x,y)
        
        # return (loss(x, y, temp=self.temperature) 
        #         + loss(x, x, temp=self.temperature) 
        #         + loss(y, y, temp=self.temperature))/3 


class NTXLoss(torch.nn.Module):
    def __init__(self, temperature=0.5):
        super(NCELoss, self).__init__()
        self.temperature = 0.5

    def forward(self, x, y):

        loss = nt_xent_loss

        return loss(x,y)


class ParticleContrastiveLoss(torch.nn.Module):
    def __init__(self, k=0.05, q1=1, q2=1):
        super(ParticleContrastiveLoss, self).__init__()
        self.k, self.q1, self.q2 = k, q1, q2
    
    def forward(self, x, y):
        # return particle_contrastive_loss(hsphere_norm(x), hsphere_norm(y))

        # return (particle_contrastive_loss(x, y) 
        #         + particle_contrastive_loss(x, x) 
        #         + particle_contrastive_loss(y, y))/3 

        return particle_contrastive_loss(x, y) 



import torch
import torch.nn
import torchvision
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

class ImageModel(pl.LightningModule):
  def__init__(self):
    super().__init___()
    
    self.save_hyperparameters()
    
    self.lr = 1e-4
    self.dropout = 0.15
    self.num_classes
    
    self.conv1 = nn.Conv2d(3, 32, 3, 1)
    self.conv2 = nn.Conv2d(32, 32, 3, 1)
    self.conv3 = nn.Conv2d(32, 64, 3, 1)
    self.conv4 = nn.Conv2d(64, 64, 3, 1)
    
    self.pool1 = torch.nn.MaxPool2d(2)
    self.pool2 = torch.nn.MaxPool2d(2)
    
    self.fc1 = nn.Linear
    
    self._dropout = torch.nn.Dropout(p=self.dropout)
    

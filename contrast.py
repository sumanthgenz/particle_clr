import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from metrics import *
from supervised import *
from data import *

wandb_logger = WandbLogger(name='supervised',project='particle_contastive_learning')

class ContrastiveModel(pl.LightningModule):

    def __init__(self):
        super(ContrastiveModel, self).__init__()

        self.feat_size = 1000

        self.resnet50 = torchvision.models.resnet50(pretrained=True)
        self.dropout = torch.nn.Dropout(p=0.10)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

        self.loss = particle_contrastive_loss()

    def forward(self, x):
        return self.resnet50(x)

    def training_step(self, batch, batch_idx):
        #x and y are two views of same anchor sample (positive pair)
        x, y = batch
        x, y = self.forward(x), self.forward(y)
        loss = self.loss(hypersphere(x), hypersphere(y))
        logs = {'loss': loss}
        return {'loss': loss, 'log': logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x, y = self.forward(x), self.forward(y)
        loss = self.loss(hypersphere(x), hypersphere(y))

        kl_div = kl_divergence(x, y)
        cos_sim = cosine_similarity(x, y)
        
        logs = {
                'val_loss': loss,
                'kl_div': kl_div,
                'cos_sim': cos_sim}

        return logs

    def test_step(self, batch, batch_idx):
        x, y = batch
        x, y = self.forward(x), self.forward(y)
        loss = self.loss(hypersphere(x), hypersphere(y))

        kl_div = kl_divergence(x, y)
        cos_sim = cosine_similarity(x, y)
        
        logs = {
                'val_loss': loss,
                'kl_div': kl_div,
                'cos_sim': cos_sim}

        return logs

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([m['val_loss'] for m in outputs]).mean()
        avg_kl_div = torch.stack([m['kl_div'] for m in outputs]).mean()
        avg_cos_sim = torch.stack([m['cos_sim'] for m in outputs]).mean()

        logs = {
        'val_loss': avg_loss,
        'val_top_1': avg_kl_div,
        'val_top_5': avg_cos_sim}

        return {'val_loss': avg_loss, 'log': logs}

    def train_dataloader(self):
        dataset = torchvision.datasets.CIFAR10(
                                root='./Desktop',
                                train=True,
                                download=True)
        return torch.utils.data.DataLoader(
                                dataset,
                                batch_size=128,
                                shuffle=True,
                                num_workers=4)

    def val_dataloader(self):
          dataset = torchvision.datasets.CIFAR10(
                                  root='./Desktop',
                                  train=False,
                                  download=True)
          return torch.utils.data.DataLoader(
                                  dataset,
                                  batch_size=128,
                                  shuffle=True,
                                  num_workers=4)

    def test_dataloader(self):
        dataset = torchvision.datasets.CIFAR10(
                                root='./Desktop',
                                train=False,
                                download=True)
        return torch.utils.data.DataLoader(
                                dataset,
                                batch_size=128,
                                shuffle=False,
                                num_workers=4)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=2e-4)

class TransferClassifier(SupervisedModel):

    def __init__(self):
        super(ClassifierModel, self).__init__()

        #set self.resnet50 to the contrastive pretrained model
        self.resnet50 = torchvision.models.resnet50(pretrained=False)

    #override forward from SupervisedModel to freeze resnet50 contrastive model
    def forward(self, x):
        with torch.no_grad():
            x = self.resnet50(x)
        x = self.dropout(self.relu(self.fc1(x)))
        # x = self.softmax(self.fc2(x))
        x = self.fc2(x)
        return x

    

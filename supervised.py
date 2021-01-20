import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from RandAugment import RandAugment


from resnet import *
from metrics import *

wandb_logger = WandbLogger(name='supervised',project='particle_contastive_learning')

#CIFAR-10

class SupervisedModel(pl.LightningModule):
    def __init__(self):
        super(SupervisedModel, self).__init__()

        self.num_classes = 10
        self.resnet_classifier = SupCEResNet()
        
        # self.feat_size = 1000

        # self.resnet50 = torchvision.models.resnet50(pretrained=False)
        # self.dropout = torch.nn.Dropout(p=0.10)
        # self.fc1 = nn.Linear(self.feat_size, 512)
        # self.fc2 = nn.Linear(512, self.num_classes)
        # self.relu = nn.ReLU()
        # self.softmax = nn.Softmax()

        self.loss = nn.CrossEntropyLoss()
        self.lr = 0.1

        #bsz*accum_grad = 128*8 = 1024
        self.bsz = 128

        #Implementation from https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
        #Implementation from https://github.com/HobbitLong/SupContrast/blob/master/main_ce.py

        #cifar 10
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)

        #cifar 100
        # mean = (0.5071, 0.4867, 0.4408)
        # std = (0.2675, 0.2565, 0.2761)

        normalize = transforms.Normalize(mean=mean, std=std)

        # self.transform_train = transforms.Compose([
        #                         transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
        #                         transforms.RandomHorizontalFlip(),
        #                         transforms.ToTensor(),
        #                         normalize,
        # ])


        self.transform_train = transforms.Compose([
                    transforms.ToTensor(),
                    normalize,
])

        self.transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    normalize,
        ])

        # N, M = 3, 4
        # self.transform_train.transforms.insert(0, RandAugment(N, M))


    def forward(self, x):
        # x = self.resnet50(x)
        # x = self.dropout(self.relu(self.fc1(x)))
        # x = self.softmax(self.fc2(x))
        # x = self.fc2(x)
        # return x
        return self.resnet_classifier(x)



    def training_step(self, batch, batch_idx):
        sample, label = batch
        logits = self.forward(sample)
        loss = self.loss(logits, label)
        logs = {'loss': loss}
        return {'loss': loss, 'log': logs}

    def validation_step(self, batch, batch_idx):
      sample, label = batch
      logits = self.forward(sample)
      loss = self.loss(logits, label)

      top_1_accuracy = compute_accuracy(logits, label, top_k=1)
      top_5_accuracy = compute_accuracy(logits, label, top_k=5)
      
      logs = {
            'val_loss': loss,
            'val_top_1': top_1_accuracy,
            'val_top_5': top_5_accuracy}

      return logs

    def test_step(self, batch, batch_idx):
      sample, label = batch
      logits = self.forward(sample)
      loss = self.loss(logits, label)

      top_1_accuracy = compute_accuracy(logits, label, top_k=1)
      top_5_accuracy = compute_accuracy(logits, label, top_k=5)

      logs = {
            'test_loss': loss,
            'test_top_1': top_1_accuracy,
            'test_top_5': top_5_accuracy}

      return logs

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([m['val_loss'] for m in outputs]).mean()
        avg_top1 = torch.stack([m['val_top_1'] for m in outputs]).mean()
        avg_top5 = torch.stack([m['val_top_5'] for m in outputs]).mean()

        logs = {
        'val_loss': avg_loss,
        'val_top_1': avg_top1,
        'val_top_5': avg_top5}

        return {'val_loss': avg_loss, 'log': logs}

    def train_dataloader(self):
        dataset = torchvision.datasets.CIFAR10(
                                root='./Desktop',
                                train=True,
                                download=True,
                                transform=self.transform_train)
        return torch.utils.data.DataLoader(
                                dataset,
                                batch_size=self.bsz,
                                shuffle=True,
                                num_workers=8)

    def val_dataloader(self):
          dataset = torchvision.datasets.CIFAR10(
                                  root='./Desktop',
                                  train=False,
                                  download=True,
                                  transform=self.transform_test)
          return torch.utils.data.DataLoader(
                                  dataset,
                                  batch_size=self.bsz,
                                  shuffle=False,
                                  num_workers=8)

    def test_dataloader(self):
        dataset = torchvision.datasets.CIFAR10(
                                root='./Desktop',
                                train=False,
                                download=True,
                                transform=self.transform_test)
        return torch.utils.data.DataLoader(
                                dataset,
                                batch_size=self.bsz,
                                shuffle=False,
                                num_workers=8)

    def configure_optimizers(self):
        # return torch.optim.Adam(self.parameters(), lr=self.lr)
    
        #to replicate supcon cross-entropy, use these hparams: https://github.com/google-research/google-research/blob/master/supcon/scripts/cross_entropy_cifar10_resnet50.sh
        optimizer = torch.optim.SGD(
                            self.parameters(), 
                            lr=self.lr, 
                            momentum=0.0, 
                            weight_decay=0,
                            nesterov=False)

        # scheduler = torch.optim.lr_scheduler.StepLR(
        #                     optimizer, 
        #                     step_size=40, 
        #                     gamma=0.1)
        # return [optimizer], [scheduler]
        return optimizer


# class ImageModel(pl.LightningModule):
#   def__init__(self):
#     super().__init___()
    
#     self.save_hyperparameters()
    
#     self.lr = 1e-4
#     self.dropout = 0.15
#     self.num_classes
    
#     self.conv1 = nn.Conv2d(3, 32, 3, 1)
#     self.conv2 = nn.Conv2d(32, 32, 3, 1)
#     self.conv3 = nn.Conv2d(32, 64, 3, 1)
#     self.conv4 = nn.Conv2d(64, 64, 3, 1)
    
#     self.pool1 = torch.nn.MaxPool2d(2)
#     self.pool2 = torch.nn.MaxPool2d(2)
    
#     self.fc1 = nn.Linear
    
#     self._dropout = torch.nn.Dropout(p=self.dropout)
    
#test push

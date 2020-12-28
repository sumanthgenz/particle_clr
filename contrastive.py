import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from RandAugment import RandAugment


from metrics import *
from supervised import *
from resnet import *
from data import *
from losses import *

wandb_logger = WandbLogger(name='supervised',project='particle_contastive_learning')

#Implementation from https://github.com/HobbitLong/SupContrast/blob/6d5a3de39070249a19c62a345eea4acb5f26c0bc/util.py#L9
class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return self.transform(x), self.transform(x)

class ContrastiveModel(pl.LightningModule):

    def __init__(self):
        super(ContrastiveModel, self).__init__()

        self.resnet_dim = 2048
        self.hidden_dim = 512
        self.feat_dim = 128

        self.lr = 1e-3
        self.resnet50 = resnet50()

        #Implementation from https://github.com/leftthomas/SimCLR/blob/cee178b6cab1efab67f8b527a0cff91c9e793f5c/model.py
        self.projection = nn.Sequential(nn.Linear(self.resnet_dim,  self.hidden_dim),
                                        nn.BatchNorm1d(self.hidden_dim),
                                        nn.ReLU(inplace=True),
                                        nn.Linear( self.hidden_dim, self.feat_dim)
        )

        self.nce = NCELoss()
        self.pcl = ParticleContrastiveLoss()

        #Implementation from https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
        #Implementation from https://github.com/HobbitLong/SupContrast/blob/master/main_ce.py

        #cifar 10
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)

        #cifar 100
        # mean = (0.5071, 0.4867, 0.4408)
        # std = (0.2675, 0.2565, 0.2761)

        normalize = transforms.Normalize(mean=mean, std=std)

        #Implementation from https://github.com/leftthomas/SimCLR/blob/master/utils.py
        self.transform_train = transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize
            ])

        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize
            ])

        #Implementation from https://github.com/HobbitLong/SupContrast/blob/master/main_ce.py
        # self.transform_train = transforms.Compose([
        #                         transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
        #                         transforms.RandomHorizontalFlip(),
        #                         transforms.ToTensor(),
        #                         normalize,
        # ])

        # self.transform_test = transforms.Compose([
        #             transforms.ToTensor(),
        #             normalize,
        # ])

        #Implementation from https://github.com/ildoonet/pytorch-randaugment
        #ImageNet
        # N, M = 2, 9

        #CIFAR-10
        # N, M = 3, 4
        # self.transform_train.transforms.insert(0, RandAugment(N, M))

    def forward(self, x):
        x = self.resnet50(x)
        x = self.projection(x)
        return hsphere_norm(x)

    def training_step(self, batch, batch_idx):
        #x and y are two views of same anchor sample (positive pair)
        x, y = batch[0][0], batch[0][1]
        x, y = self.forward(x), self.forward(y)
        loss = self.pcl(x, y)
        # loss = self.nce(x, y)

        logs = {'loss': loss}
        return {'loss': loss, 'log': logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch[0][0], batch[0][1]
        x, y = self.forward(x), self.forward(y)
        pcl_loss = self.pcl(x, y)
        nce_loss = self.nce(x, y)

        # kl_div = kl_divergence(x, y)
        # cos_sim = cosine_similarity(x, y)
        
        # logs = {
        #         'val_nce_loss': loss,
        #         'kl_div': kl_div,
        #         'cos_sim': cos_sim}

        logs = {'val_pcl_loss': pcl_loss,
                'val_nce_loss': nce_loss
        }

        return logs

    def test_step(self, batch, batch_idx):
        x, y = batch[0][0], batch[0][1]
        x, y = self.forward(x), self.forward(y)
        pcl_loss = self.pcl(x, y)

        # kl_div = kl_divergence(x, y)
        # cos_sim = cosine_similarity(x, y)
        
        # logs = {
        #         'val_nce_loss': loss,
        #         'kl_div': kl_div,
        #         'cos_sim': cos_sim}

        logs = {'val_pcl_loss': pcl_loss}

        return logs

    def validation_epoch_end(self, outputs):
        avg_pcl_loss = torch.stack([m['val_pcl_loss'] for m in outputs]).mean()
        avg_nce_loss = torch.stack([m['val_nce_loss'] for m in outputs]).mean()

        # avg_kl_div = torch.stack([m['kl_div'] for m in outputs]).mean()
        # avg_cos_sim = torch.stack([m['cos_sim'] for m in outputs]).mean()

        # logs = {
        # 'val_loss': avg_loss,
        # 'val_top_1': avg_kl_div,
        # 'val_top_5': avg_cos_sim}

        logs =  {'val_pcl_loss': avg_pcl_loss,
                'val_nce_loss': avg_nce_loss
        }

        return {'val_pcl_loss': avg_pcl_loss, 'log': logs}

    def train_dataloader(self):
        dataset = torchvision.datasets.CIFAR10(
                                root='./Desktop',
                                train=True,
                                transform=TwoCropTransform(self.transform_train),
                                download=True)
        return torch.utils.data.DataLoader(
                                dataset,
                                batch_size=128,
                                shuffle=True,
                                num_workers=8)

    def val_dataloader(self):
          dataset = torchvision.datasets.CIFAR10(
                                  root='./Desktop',
                                  train=False,
                                  transform=TwoCropTransform(self.transform_test),
                                  download=True)
          return torch.utils.data.DataLoader(
                                  dataset,
                                  batch_size=128,
                                  shuffle=True,
                                  num_workers=8)

    def test_dataloader(self):
        dataset = torchvision.datasets.CIFAR10(
                                root='./Desktop',
                                train=False,
                                transform=TwoCropTransform(self.transform_test),
                                download=True)
        return torch.utils.data.DataLoader(
                                dataset,
                                batch_size=128,
                                shuffle=False,
                                num_workers=8)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
                            self.parameters(), 
                            lr=self.lr, 
                            weight_decay=0)

        # optimizer = torch.optim.RMSprop(
        #             self.parameters(), 
        #             lr=self.lr, 
        #             momentum=0.0, 
        #             weight_decay=0)

        # scheduler = torch.optim.lr_scheduler.StepLR(
        #                     optimizer, 
        #                     step_size=40, 
        #                     gamma=0.1)
        # return [optimizer], [scheduler]
        return optimizer

class LinearClassifier(SupervisedModel):

    def __init__(self):
        super(LinearClassifier, self).__init__()

        #set self.resnet50 to the contrastive pretrained model
        self.num_classes = 10
        self.path = 'Desktop/info_nce_epoch=28.ckpt'
        self.model = ContrastiveModel()
        self.model.load_state_dict(torch.load(self.path), strict=False)
        self.fc1 = nn.Linear(2048, self.num_classes)
        self.lr = 1e-4

    #override forward from SupervisedModel to freeze resnet50 contrastive model
    def forward(self, x):
        with torch.no_grad():
            x = self.model(x)
        x = self.fc1(x)
        return x

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

class ImageData(Dataset):
    
        def __init__(self, csvType):
            self.dir = "/Desktop/{}".format("cifar-10")
            self.num_classes = 10
            self.wav_paths = self.load_data()
            self.transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            self.transform_test = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            
        def load_data(self):
            wav_paths = []
            for path in glob.glob(f'{self.dir}/**/*.wav'):
                wav_paths.append(path)
            return wav_paths
        
        def __len__(self):
            return len(self.wav_paths)

        def getNumClasses(self):
            return self.num_classes

        def __getitem__(self, idx): 
            return wav_paths[idx], self.transforms.forward(wav_paths[idx])
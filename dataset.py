"""
source:
https://www.kaggle.com/pankajj/fashion-mnist-with-pytorch-93-accuracy
"""
import numpy as np
import pandas as pd
import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
import albumentations as A
from albumentations.pytorch import ToTensorV2

class AlbumentationsTransform:
    def __init__(self):
        self.img_transforms = A.Compose([ 
                                A.HorizontalFlip(p=0.5),
                                A.VerticalFlip(p=0.5),
                                ToTensorV2(),
                            ])
    def __call__(self, img):
        img = np.array(img)
        return self.img_transforms(image = img).copy()

class AlbumentationsTransformTest:
    def __init__(self):
        self.img_transforms = A.Compose([ 
                                ToTensorV2(),
                            ])
    def __call__(self, img):
        img = np.array(img)
        return self.img_transforms(image = img).copy()


class FashionMnist(Dataset):
    def __init__(self, data_as_pandas_dataframe, transform=None):
        """
        We will focus on detecting shoes. Labels 5, 7 and 9 (Sandal, Sneaker and Ankle boot)
        should be treated as positive and all others (such as Trouser, Dress and so on)
        should be negative.
        """
        self.label_map = {
            0: 0,
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 1,
            6: 0,
            7: 1,
            8: 0,
            9: 1,
        }
        self.fashion_MNIST = list(data_as_pandas_dataframe.values)
        self.transform = transform
        label = []
        image = []
        for i in self.fashion_MNIST:
            label.append(self.label_map[i[0]])
            image.append(i[1:])
        self.labels = np.asarray(label)
        self.images = np.asarray(image).reshape(-1, 28, 28, 1).astype('float32')
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        label = self.labels[index]
        image = self.images[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class ImbalancedDatasetSampler(Sampler):
    def __init__(self, dataset):
        self.indices = list(range(len(dataset)))
        self.num_samples = len(self.indices)
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        return dataset[idx][1]

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples
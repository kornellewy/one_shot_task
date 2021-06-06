"""
source:
https://github.com/kornellewy/youtube-collection
https://www.youtube.com/watch?v=QYiAumP7HtE&t=2s
https://www.kaggle.com/pankajj/fashion-mnist-with-pytorch-93-accuracy
"""

import torch
import cv2
import os
import torch.nn as nn
import numpy as np
import torchvision
import pandas as pd
from torchvision import datasets, models, transforms
from torch.utils.data.sampler import Sampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from sklearn.metrics import accuracy_score, confusion_matrix
from matplotlib import pyplot as plt
from pathlib import Path

from dataset import AlbumentationsTransform, ImbalancedDatasetSampler, FashionMnist
from model_ver001 import FashionCNNVer001

class ClassificationTrainer(pl.LightningModule):
    def __init__(self, hparams = {
            'epochs_num': 100,
            'batch_size': 32,
            'lr': 0.001,},
        folders_structure = {
            "models_folder": str(Path(__file__).parent / "models"),
            "confusion_matrix_folder": str(Path(__file__).parent / "confusion_matrix"),
            "test_img_folder": str(Path(__file__).parent / "test_img_folder"),
            "metadata_json_folder": str(Path(__file__).parent / "metadata_json")
        }):
            super().__init__()
            self._hparams = hparams
            
            self.img_transform = transforms.Compose([AlbumentationsTransform()])
            self.criterion = nn.CrossEntropyLoss()

            self.model = FashionCNNVer001()

            self.folders_structure = folders_structure
            self._create_dir_structure()
            self.test_losses = []
            self.test_true = []
            self.test_pred = []

    def _create_dir_structure(self):
        for _, path in self.folders_structure.items():
            os.makedirs(path, exist_ok=True)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_nb):
        x,y = batch
        y = y.long()
        if isinstance(x, dict):
            x = x['image']
        pred = self(x)
        loss = self.criterion(pred, y)
        self.log('train_loss', loss)
        self.log('avg_train_loss', loss ,on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_nb):
        x,y = batch
        y = y.long()
        if isinstance(x, dict):
            x = x['image']
        pred = self(x)
        loss = self.criterion(pred, y)
        self.log('test_loss_per_batch', loss)

        ps = torch.exp(pred)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == y.view(*top_class.shape)
        accuracy = torch.mean(equals.type(torch.FloatTensor)).item()
        self.log('test_accuracy_per_batch', accuracy)

        self.test_losses.append(loss.item())
        self.test_true += y.tolist()
        self.test_pred += top_class.tolist()
        return loss

    def test_epoch_end(self, outputs):
        self.test_pred = [ i[0] for i in self.test_pred ]

        loss = sum(self.test_losses) / len(self.test_losses)
        self.log('test_loss_per_epoch', loss)

        accuracy = accuracy_score(self.test_true, self.test_pred)
        self.log('test_accuracy_per_epoch', accuracy)

        self.save_model(loss=loss, acc=accuracy, mode='test')
        return 

    def save_model(self, loss=0.0, acc=0.0, mode='valid'):
        model_name = mode+'_'+'loss_' + str(round(loss, 4)) + '_accuracy_' + str(round(acc, 4)) + '.pth'
        model_save_path = os.path.join(self.folders_structure['models_folder'], model_name)
        torch.save(self.model, model_save_path)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self._hparams['lr'])


if __name__ == '__main__':
    torch.cuda.empty_cache()
    hparams = hparams = {
            'epochs_num': 10,
            'batch_size': 256,
            'lr': 0.0001,
        }
    img_transform = transforms.Compose([AlbumentationsTransform()])

    train_set_path = 'Fashion_MNIST/fashion-mnist_train.csv'
    train_set = pd.read_csv(train_set_path)
    train_set = FashionMnist(train_set, img_transform)

    test_set_path = 'Fashion_MNIST/fashion-mnist_train.csv'
    test_set = pd.read_csv(test_set_path)
    test_set = FashionMnist(test_set, img_transform)

    train_dataloader = torch.utils.data.DataLoader(train_set,
                                    sampler=ImbalancedDatasetSampler(train_set),
                                    batch_size=hparams['batch_size'])
    test_dataloader = torch.utils.data.DataLoader(test_set,
                                    sampler=ImbalancedDatasetSampler(test_set),
                                    batch_size=hparams['batch_size'])

    model = ClassificationTrainer(hparams=hparams)
    trainer = Trainer(gpus=1, benchmark=True, precision=16, 
                    max_epochs=10, check_val_every_n_epoch=1,
                    # resume_from_checkpoint=checkpoint_path 
                    )
    trainer.fit(model, train_dataloader, test_dataloader)
    trainer.test(model, test_dataloader)
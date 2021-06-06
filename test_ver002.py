"""
source:
https://github.com/kornellewy/youtube-collection
https://www.youtube.com/watch?v=QYiAumP7HtE&t=2s
https://www.kaggle.com/pankajj/fashion-mnist-with-pytorch-93-accuracy
https://towardsdatascience.com/one-shot-learning-with-siamese-networks-using-keras-17f34e75bb3d
https://towardsdatascience.com/building-a-one-shot-learning-network-with-pytorch-d1c3a5fafa4a
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

from model_ver002 import FashionCNNVer002
from dataset import AlbumentationsTransform, AlbumentationsTransformTest

def load_images(path):
    images = []
    valid_images = [".jpeg", ".jpg", '.png']
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        images.append(os.path.join(path, f))
    return images

def main_zad2():
    device = torch.device("cpu")
    base_model_path = 'models/test_loss_0.005_accuracy_0.9984.pth'
    base_model = torch.load(base_model_path).to(device)
    model = FashionCNNVer002(base_model = base_model).to(device)

    img_transforms = transforms.Compose([AlbumentationsTransformTest()])

    flats_example_image_path = 'Shoes/train/Flats/646.png'
    flats_img = cv2.imread(flats_example_image_path, 0)
    flats_img = img_transforms(flats_img)['image'].to(torch.float32)
    flats_img = flats_img.unsqueeze(0)
    flats_img = flats_img.to(device)

    heels_example_image_path = 'Shoes/train/Heels/746.png'
    heels_img = cv2.imread(heels_example_image_path, 0)
    heels_img = img_transforms(heels_img)['image'].to(torch.float32)
    heels_img = heels_img.unsqueeze(0)
    heels_img = heels_img.to(device)

    flats_dir_path = 'Shoes/test/Flats'
    flats_images_paths = load_images(flats_dir_path)
    flats_pred_list = []
    for test_image_path in flats_images_paths:
        test_img = cv2.imread(test_image_path, 0)
        test_img = img_transforms(test_img)['image'].to(torch.float32)
        test_img = test_img.unsqueeze(0)
        test_img = test_img.to(device)
        pred = model.forward(flats_img, heels_img, test_img).item()
        flats_pred_list.append(pred)
    flats_true_list = [ 0 for i in range(len(flats_pred_list)) ]
    flats_acc = accuracy_score(flats_true_list, flats_pred_list)
    print('flats_acc: ', flats_acc)

    heels_dir_path = 'Shoes/test/Heels'
    heels_images_paths = load_images(heels_dir_path)
    heels_pred_list = []
    for test_image_path in heels_images_paths:
        test_img = cv2.imread(test_image_path, 0)
        test_img = img_transforms(test_img)['image'].to(torch.float32)
        test_img = test_img.unsqueeze(0)
        test_img = test_img.to(device)
        pred = model.forward(flats_img, heels_img, test_img).item()
        heels_pred_list.append(pred)
    heels_true_list = [ 0 for i in range(len(heels_pred_list)) ]
    heels_acc = accuracy_score(heels_true_list, heels_pred_list)
    print('heels_acc: ', heels_acc)

    # makro acc
    true_list = flats_true_list + heels_true_list
    pred_list = flats_pred_list + heels_pred_list
    macro_acc = accuracy_score(true_list, pred_list)
    print('macro_acc: ', macro_acc)

if __name__ == '__main__':
    main_zad2()
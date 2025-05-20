import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch import optim
from torch import tensor
from torch.functional import F
from torchvision import transforms, datasets
from torchvision import models
from torch.utils.data.sampler import WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, root_mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

from skimage.morphology import skeletonize
from itertools import product

import pandas as pd
import numpy as np
import os
from PIL import Image
import copy
import cv2
import time
import shutil
import random
import h5py
from Helpers import Params

from matplotlib import pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class HandWritingDataset(datasets.ImageFolder):
    def __init__(
        self,
        root,
        transform: transforms.Compose = None,
        label_transofrm=None,
        augument: transforms.Compose = None
    ):
        if augument is not None:
            transform = transforms.Compose([
                augument,
                transform
            ])

        super(HandWritingDataset, self).__init__(root, transform=transform)
        self.classes = [''.join([i if i != '_' else '' for i in word])
                        for word in self.classes]
        self.label_transform = label_transofrm

    def __getitem__(self, index):
        image, label = super(HandWritingDataset, self).__getitem__(index)
        if self.label_transform is not None:
            label = tensor(self.label_transform(self.classes[label]))
        return image, label
    
    def create_sampler(self, epsilon: float = 0.004):
        """
        epsilon: how strongly flatten the distribution? 0 means flat, the greater the more similar to original distribution, after 0.01 there is little difference
        """
        df = pd.DataFrame(self.samples, columns=['filename', 'class'])
        df['label'] = df['class'].apply(lambda x: ''.join(
            [i for i in self.classes[x] if i != ' ']))
        df['length'] = df['label'].apply(len)
        length_df = df['length'].value_counts().sort_index().to_dict()
        class_count = df['class'].value_counts().sort_index()
        df['class_length'] = df['length'].map(length_df)
        df['class_count'] = df['class'].apply(lambda x: class_count.iloc[x])
        df['result'] = 1.0 / df['class_length'] + epsilon
        return WeightedRandomSampler(df['result'].values, len(df['result']))


class H5Dataset:
    def __init__(self, file_path: str, num_epochs: int):
        """
        Args:
            num_epochs (int): Number of epochs to split the dataset into.
        """
        self.file_path = file_path
        with h5py.File(file_path, 'r') as h5f:
            h5_size = len(h5f['images'])
            self.num_epochs = num_epochs
            self.epoch_size = h5_size // num_epochs
            self.current_epoch = 0
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, index):
        with h5py.File(self.file_path, 'r') as h5f:
            image = h5f['images'][self.current_epoch * self.epoch_size + index]
            image = self.transform(image)
            image = image.permute(1, 2, 0)
            image = torch.stack([image[0]] * 3, dim=0)

            label = h5f['labels'][self.current_epoch * self.epoch_size + index]
            label = label[:np.argmax(label == 0)] if 0 in label else label
            label = torch.tensor(label, dtype=torch.long)

        return image, label

    def next_epoch(self):
        self.current_epoch = (self.current_epoch + 1) % self.num_epochs

    def create_h5_sampler(self, epsilon: float = 0.004):
        """
        epsilon: how strongly flatten the distribution? 0 means flat, the greater the more similar to original distribution, after 0.01 there is little difference
        """
        df = pd.DataFrame({'index': range(
            self.current_epoch * self.epoch_size, (self.current_epoch + 1) * self.epoch_size)})
        with h5py.File(self.file_path, 'r') as h5f:
            labels = h5f['labels'][self.current_epoch *
                                   self.epoch_size: (self.current_epoch + 1) * self.epoch_size]
            df['length'] = (labels != 0).sum(axis=1)

        length_df = df['length'].value_counts().sort_index()
        length_dict = length_df.to_dict()

        df['class_length'] = df['length'].map(length_dict)
        df['result'] = 1.0 / df['class_length'] + epsilon

        return WeightedRandomSampler(df['result'].values, len(df['result']))


def create_h5_dataset(size: int = 10):
    if os.path.exists('train_data.h5'):
        os.remove('train_data.h5')

    h5_augument = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomRotation(15, expand=True, fill=(255,)),
        transforms.RandomAffine(0, translate=(0.05, 0.05), fill=(255,)),
        transforms.RandomPerspective(distortion_scale=0.5, p=0.5, fill=(255,)),
        transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.Resize((64, 128)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: 1.0 - x),
    ])

    h5_dataset = HandWritingDataset(
        root='words_data/train', transform=h5_augument, label_transofrm=Params.encode_string)

    start_time = time.time()

    with h5py.File('train_data.h5', 'a') as h5f:
        h5f.create_dataset('images', shape=(
            size * len(h5_dataset), 1, 64, 128), dtype=np.uint8)
        h5f.create_dataset('labels', shape=(
            size * len(h5_dataset), Params.MAX_LEN), dtype=np.uint8)

        for i in range(len(h5_dataset)):
            for j in range(size):
                image, label = h5_dataset[i]
                image = image.numpy() * 255
                label = np.pad(label.numpy(), (0, Params.MAX_LEN -
                               len(label)), 'constant', constant_values=0)
                h5f['images'][i * size + j] = image
                h5f['labels'][i * size + j] = label
                if all(label == 0):
                    print(label)

            remaining_time = (time.time() - start_time) * (len(h5_dataset)
                                                           * size - i * size - j) / (i * size + j + 1)
            print(
                f'\r {i * size + j} / {len(h5_dataset) * size}    {int(remaining_time / 60)}:{int(remaining_time % 60)}  ', end='')

# create_h5_dataset(1)

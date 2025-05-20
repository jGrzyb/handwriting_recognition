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

from matplotlib import pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class Params:
    MAX_LEN = 12
    vocab = " abcdefghijklmnopqrstuvwxyz"  # 1-indexed
    char_to_index = {char: i for i, char in enumerate(vocab)}
    index_to_char = {i: char for i, char in enumerate(vocab)}

    @staticmethod
    def encode_string(s: str):
        return [Params.char_to_index[char] for char in s]

    @staticmethod
    def decode_string(encoded: list[int]):
        return ''.join([Params.index_to_char[i] for i in encoded if i != 0])
    

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

        for j in range(size):
            for i in range(len(h5_dataset)):
                image, label = h5_dataset[i]
                image = image.numpy() * 255
                label = np.pad(label.numpy(), (0, Params.MAX_LEN -
                           len(label)), 'constant', constant_values=0)
                h5f['images'][j * len(h5_dataset) + i] = image
                h5f['labels'][j * len(h5_dataset) + i] = label
                # print(label)

                remaining_time = (time.time() - start_time) * (len(h5_dataset)
                                       * size - j * len(h5_dataset) - i) / (j * len(h5_dataset) + i + 1)
                print(
                f'\r {j * len(h5_dataset) + i} / {len(h5_dataset) * size}    {int(remaining_time / 60)}:{int(remaining_time % 60)}  ', end='')

create_h5_dataset(20)

# with h5py.File('train_data.h5', 'r') as h5f:
#     for i in range(2800):
#         print(h5f['images'][i].shape, h5f['labels'][i])
        

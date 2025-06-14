from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
from utils.params import Params
import torch
from torch import tensor
from torchvision import transforms
import h5py
from torch.utils.data.sampler import WeightedRandomSampler
import pandas as pd

class HandWritingDataset(Dataset):
    def __init__(self, root, transform=None, label_transform=None):
        self.root = root
        self.transform = transform
        self.label_transform = label_transform
        self.images = []
        self.labels = []
        self.load_data()

    def load_data(self):
        for label in os.listdir(self.root):
            label_path = os.path.join(self.root, label)
            if os.path.isdir(label_path):
                for img_file in os.listdir(label_path):
                    img_path = os.path.join(label_path, img_file)
                    self.images.append(img_path)
                    self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        if self.label_transform:
            label = self.label_transform(label)

        return image, label

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
        label = torch.cat((tensor([27]), label, tensor([28])))

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
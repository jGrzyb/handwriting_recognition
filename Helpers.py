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


class EarlyStopping:
    def __init__(self, patience=5, save_path='best_model.pth'):
        """
        Args:
            patience (int): How many epochs to wait after the last improvement.
            save_path (str): Path to save the best model.
        """
        self.patience = patience
        self.save_path = save_path
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss: float, model: nn.Module):
        """
        Args:
            val_loss (float): Validation loss for the current epoch.
            model (nn.Module): Model to save if it has the best performance so far.
        """
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, model: nn.Module):
        """Saves the current best model to the specified path."""
        torch.save(model.state_dict(), self.save_path)


class ProgressBarWithLoss:
    def __init__(self, total_epochs, total_batches):
        self.total_epochs = total_epochs
        self.total_batches = total_batches
        self.bar_length = 30
        self.start_time = time.time()

    def update(self, epoch, batch_idx, train_loss, val_loss=None):
        elapsed_time = time.time() - self.start_time
        progress = (batch_idx + 1) / self.total_batches
        filled_length = int(self.bar_length * progress)
        bar = '=' * filled_length + '-' * (self.bar_length - filled_length)
        val_str = f"Val Loss: {val_loss:.4f}\n" if val_loss else ''
        print(f"Train Loss: {train_loss:.4f}   {val_str}", end='')


class Validator:
    def __init__(self, model: nn.Module, val_loader: DataLoader, criterion: nn.Module):
        self.model = model
        self.val_loader = val_loader
        self.criterion = criterion

    def validate(self):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = self.model(images)

                batch_size = images.size(0)
                input_lengths = torch.full(
                    size=(batch_size,), fill_value=outputs.size(0), dtype=torch.long).to(device)
                target_lengths = torch.tensor(
                    [len(seq[seq != 0]) for seq in labels], dtype=torch.long).to(device)

                loss = self.criterion(
                    outputs, labels, input_lengths, target_lengths)
                val_loss += loss.item()

        val_loss /= len(self.val_loader)
        return val_loss


class History:
    def __init__(self):
        self.history = {'train_loss': [], 'val_loss': []}

    def add_epoch(self, train_loss, val_loss):
        """
        Adds the training and validation loss for an epoch.

        Args:
            train_loss (float): Training loss for the epoch.
            val_loss (float): Validation loss for the epoch.
        """
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)

    def plot(self):
        """
        Plots the training and validation loss over epochs.
        """
        plt.figure(figsize=(10, 5))
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.ylim((0, 4))
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.show()


def collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images, dim=0)  # Stack images into a single tensor
    lengths = torch.tensor([len(label) for label in labels])  # Original lengths of labels
    labels = pad_sequence(labels, batch_first=True, padding_value=0)  # Pad labels
    return images, labels


def predict(model: nn.Module, dataloader: DataLoader, amount: int = 100):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            outputs = model(images).squeeze(1)
            labels = labels.squeeze(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend([Params.decode_string(preds.cpu().numpy())])
            all_labels.extend([Params.decode_string(labels.cpu().numpy())])
            if len(all_preds) >= amount:
                break
    return list(zip(all_preds, all_labels))


def train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, optimizer: optim.Optimizer, criterion: nn.Module, epochs: int, early_stopping: EarlyStopping, history: History, scheduler: optim.lr_scheduler = None):
    validator = Validator(model, val_loader, criterion)

    for epoch in range(epochs):
        progress_bar = ProgressBarWithLoss(epochs, len(train_loader))
        model.train()
        train_loss = 0.0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            batch_size = images.size(0)
            input_lengths = torch.full(
                size=(batch_size,), fill_value=outputs.size(0), dtype=torch.long).to(device)
            target_lengths = torch.tensor(
                [len(seq[seq != 0]) for seq in labels], dtype=torch.long).to(device)

            loss = criterion(outputs, labels, input_lengths, target_lengths)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            progress_bar.update(epoch + 1, batch_idx,
                                train_loss / (batch_idx + 1))
        if scheduler is not None:
            scheduler.step()

        train_loss /= len(train_loader)
        val_loss = validator.validate()

        progress_bar.update(epoch + 1, len(train_loader) -
                            1, train_loss, val_loss)

        history.add_epoch(train_loss, val_loss)

        if early_stopping:
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

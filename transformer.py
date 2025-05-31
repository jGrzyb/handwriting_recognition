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
from Helpers import Params, collate_fn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
from matplotlib import pyplot as plt


class ProgressBar:
    def __init__(self, total: int):
        self.total = total
        self.current = 0
        self.last_progress = 0
        self.start_time = time.time()

    def update(self, current: int, epochs: str):
        self.current = current
        progress = (self.current / self.total) * 100
        if int(progress) > self.last_progress:
            elapsed_time = time.time() - self.start_time
            print(
                f'\rEpoch: {epochs.rjust(7)} {str(int(progress)).rjust(3)}% | Elapsed: {str(int(elapsed_time)).rjust(3)}s', end='')
            self.last_progress = int(progress)

class EarlyStopping:
    def __init__(self, patience=5, delta=0, path='checkpoint.pt', verbose=False):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(
                    f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

class TrainingHistory:
    def __init__(self):
        self.history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }

    def update(self, train_loss: float, train_accuracy: float, val_loss: float, val_accuracy: float):
        self.history['train_loss'].append(train_loss)
        self.history['train_accuracy'].append(train_accuracy)
        self.history['val_loss'].append(val_loss)
        self.history['val_accuracy'].append(val_accuracy)

    def get_history(self):
        return self.history

    def plot(self):
        epochs = range(1, len(self.history['train_loss']) + 1)
        plt.figure(figsize=(12, 6))

        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.history['train_loss'], label='Train Loss')
        plt.plot(epochs, self.history['val_loss'], label='Validation Loss')
        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(
            epochs, self.history['train_accuracy'], label='Train Accuracy')
        plt.plot(epochs, self.history['val_accuracy'],
                 label='Validation Accuracy')
        plt.title('Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.show()

class MetricsTracker:
    def __init__(self):
        self.losses = []
        self.accuracies = []

    def update(self, loss: float, accuracy: float):
        self.losses.append(loss)
        self.accuracies.append(accuracy)

    def get_average_loss(self):
        return sum(self.losses) / len(self.losses) if self.losses else 0.0

    def get_average_accuracy(self):
        return sum(self.accuracies) / len(self.accuracies) if self.accuracies else 0.0
    
class Validator:
    def __init__(self, model: nn.Module, criterion: nn.CrossEntropyLoss, device: torch.device):
        self.model = model
        self.criterion = criterion
        self.device = device

    def validate(self, val_loader: DataLoader) -> tuple:
        self.model.eval()
        metrics_tracker = MetricsTracker()

        with torch.no_grad():
            for image, midi in val_loader:
                image: torch.Tensor = image.to(self.device)
                midi: torch.Tensor = midi.to(self.device)

                input_tokens: torch.Tensor = midi[:, :-1]
                target_tokens: torch.Tensor = midi[:, 1:]
                target_tokens = target_tokens.reshape(-1)

                output: torch.Tensor = self.model(image, input_tokens)
                output = output.reshape(-1, output.shape[-1])

                loss: torch.Tensor = self.criterion(output, target_tokens)
                accuracy = (output.argmax(dim=1) ==
                            target_tokens).float().mean().item()

                metrics_tracker.update(loss.item(), accuracy)

        avg_loss = metrics_tracker.get_average_loss()
        avg_accuracy = metrics_tracker.get_average_accuracy()

        return avg_loss, avg_accuracy
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model).to(device)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor):
        x = x + self.pe[:, :x.size(1)]
        return x
    
class ValidatorWithOutputs:
    def __init__(self, model: nn.Module, criterion: nn.CrossEntropyLoss, device: torch.device):
        self.model = model
        self.criterion = criterion
        self.device = device

    def validate(self, val_loader: DataLoader) -> tuple:
        self.model.eval()
        outputs = []

        with torch.no_grad():
            for image, midi in val_loader:
                image: torch.Tensor = image.to(self.device)
                midi: torch.Tensor = midi.to(self.device)

                input_tokens: torch.Tensor = midi[:, :-1]
                target_tokens: torch.Tensor = midi[:, 1:]
                target_tokens = target_tokens.reshape(-1)

                output: torch.Tensor = self.model(image, input_tokens)
                output = output.reshape(-1, output.shape[-1])

                predicted = output.argmax(dim=1).cpu().tolist()
                expected = target_tokens.cpu().tolist()

                outputs.append((predicted, expected))

        return outputs
    
def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: torch.device = device,
    history: TrainingHistory = None,
    early_stopping: EarlyStopping = None
):
    for epoch in range(epochs):
        progress_bar = ProgressBar(len(train_loader))
        metrics_tracker = MetricsTracker()
        validator = Validator(model, criterion, device)
        model.train()

        for i, (image, word) in enumerate(train_loader):
            image: torch.Tensor = image.to(device)
            word: torch.Tensor = word.to(device)

            input_tokens = word[:, :-1]
            target_tokens = word[:, 1:]
            # print(input_tokens)
            # print(target_tokens)
            target_tokens = target_tokens.reshape(-1)

            output: torch.Tensor = model(image, input_tokens)
            output = output.reshape(-1, output.shape[-1])

            loss: torch.Tensor = criterion(output, target_tokens)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            accuracy = (output.argmax(dim=1) == target_tokens).float().mean().item()
            metrics_tracker.update(loss.item(), accuracy)

            progress_bar.update(i + 1, f'{epoch + 1}/{epochs}')

        print(
            f' | loss: {metrics_tracker.get_average_loss():.4f} - acc: {metrics_tracker.get_average_accuracy():.4f}', end='')
        val_loss, val_acc = validator.validate(val_loader)
        print(f' | val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}')

        if history:
            history.update(metrics_tracker.get_average_loss(
            ), metrics_tracker.get_average_accuracy(), val_loss, val_acc)

        if early_stopping:
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

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
        label = torch.cat((tensor([27]), label, tensor([28])))
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
    
class HandwritingTransformer(nn.Module):
    def __init__(
        self,
        input_size: int,
        vocab_size: int,
        d_model: int = 128,
        nhead_en: int = 1,
        num_layers_en: int = 1,
        nhead_de: int = 1,
        num_layers_de: int = 1,
        dropout: float = 0.2  # Added dropout parameter
    ):
        super(HandwritingTransformer, self).__init__()

        self.cnn = models.mobilenet_v2(pretrained=True).features[:4]
        for param in self.cnn.parameters():
            param.requires_grad = False

        self.input_size = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, 200)

        # Add LayerNorm and Dropout after input_size
        self.input_norm = nn.LayerNorm(d_model)
        self.input_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead_en, dropout=dropout, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers_en)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model, nhead_de, dropout=dropout, norm_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers_de)

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_decoder = PositionalEncoding(d_model, 20)

        # Add LayerNorm and Dropout after embedding
        self.embedding_norm = nn.LayerNorm(d_model)
        self.embedding_dropout = nn.Dropout(0.5)

        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor):
        src = torch.stack([src[:, 0]] * 3, dim=1)
        src = self.cnn(src)

        src = src.flatten(1, 2)
        src = src.permute(0, 2, 1)
        src = self.input_size(src)

        # Apply LayerNorm and Dropout after input_size
        src = self.input_norm(src)
        src = self.input_dropout(src)

        src = self.pos_encoder(src)
        src = src.permute(1, 0, 2)
        memory = self.encoder(src)

        tgt = self.embedding(tgt)

        # Apply LayerNorm and Dropout after embedding
        tgt = self.embedding_norm(tgt)
        tgt = self.embedding_dropout(tgt)

        tgt = self.pos_decoder(tgt)
        tgt = tgt.permute(1, 0, 2)

        tgt_mask = self.generate_square_subsequent_mask(
            tgt.size(0)).to(tgt.device)

        output: torch.Tensor = self.decoder(tgt, memory, tgt_mask=tgt_mask)
        output = self.output(output)
        output = output.permute(1, 0, 2)
        return output

    def generate_square_subsequent_mask(self, sz):
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)
    

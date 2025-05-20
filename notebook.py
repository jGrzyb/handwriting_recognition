# %% [markdown]
# # Instruction
# 
# - Download datasets from links below and run commented code in **Data** section, that will prepare folder with images.
# - Fold everything for better understanding

# %% [markdown]
# # Links
# 
# Texts: [https://www.kaggle.com/datasets/naderabdalghani/iam-handwritten-forms-dataset](https://www.kaggle.com/datasets/naderabdalghani/iam-handwritten-forms-dataset)
# 
# Words: [https://www.kaggle.com/datasets/nibinv23/iam-handwriting-word-database](https://www.kaggle.com/datasets/nibinv23/iam-handwriting-word-database)

# %% [markdown]
# # Imports

# %%
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

# %% [markdown]
# # Params and Helpers

# %%
class Params:
    MAX_LEN = 12
    vocab = " abcdefghijklmnopqrstuvwxyz" # 1-indexed
    char_to_index = {char: i for i, char in enumerate(vocab)}
    index_to_char = {i: char for i, char in enumerate(vocab)}

    @staticmethod
    def encode_string(s: str):
        return [Params.char_to_index[char] for char in s]

    @staticmethod
    def decode_string(encoded: list[int]):
        return ''.join([Params.index_to_char[i] for i in encoded if i != 0])

# %%
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

# %%
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
        print(f"\rEpoch {epoch}/{self.total_epochs} [{bar}] {progress * 100:.2f}%   {elapsed_time:.2f}s   "
              f"Train Loss: {train_loss:.4f}   {f'Val Loss: {val_loss:.4f}\n' if val_loss else ''}", end='')

# %%
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

                loss = self.criterion(outputs, labels, input_lengths, target_lengths)
                val_loss += loss.item()

        val_loss /= len(self.val_loader)
        return val_loss

# %%
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

# %% [markdown]
# # Data

# %%
# # Get list of images and labels
# with open('iam_words/words.txt', 'r') as f:
#     words = f.readlines()
# words = [word.strip() for word in words]
# words = words[18:-1]
# words = [w for w in words if ' err ' not in w]
# words = [[w.split(' ')[0], w.split(' ')[-1]] for w in words]
# words = [
#     [f'iam_words/words/{w.split('-')[0]}/{w.split('-')[0]}-{w.split('-')[1]}/{w}.png', y] for w, y in words]
# df = pd.DataFrame(words, columns=['filename', 'word'])
# df = df[df['filename'].apply(os.path.exists)]



# # Filter out invalid images
# valid_rows = []
# for i, (path, label) in df.iterrows():
#     try:
#         with Image.open(path) as img:
#             img.verify()  # Verify that the file is a valid image
#         valid_rows.append((path, label))
#     except Exception as e:
#         print(f"Skipping file {path} due to error: {e}")
#     if i % (len(df) //100 + 1) == 0:
#         print('.', end='')
# print('\n')

# df = pd.DataFrame(valid_rows, columns=df.columns)
# df = df[df['word'].apply(lambda x: all(char in Params.vocab[1:]
#                          for char in x) and len(x) <= Params.MAX_LEN)]



# # Split data into sets convinient for torch dataset
# train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
# test_df, val_df = train_test_split(test_df, test_size=0.5, random_state=42)
# train_df = train_df.reset_index(drop=True)
# val_df = val_df.reset_index(drop=True)
# test_df = test_df.reset_index(drop=True)



# # Save images in convinient for torch way
# shutil.rmtree('words_data', ignore_errors=True)

# for i, (im, label) in train_df.iterrows():
#     new_path = f'words_data/train/{label.ljust(Params.MAX_LEN, "_")}'
#     if not os.path.exists(new_path):
#         os.makedirs(new_path)
#     im = Image.open(im)
#     im.save(f'{new_path}/{i}.png')
#     if i % (len(train_df) // 100 + 1) == 0:
#         print('.', end='')
# print()

# for i, (im, label) in val_df.iterrows():
#     new_path = f'words_data/val/{label.ljust(Params.MAX_LEN, "_")}'
#     if not os.path.exists(new_path):
#         os.makedirs(new_path)
#     im = Image.open(im)
#     im.save(f'{new_path}/{i}.png')
#     if i % (len(val_df) // 100 + 1) == 0:
#         print('.', end='')
# print()


# for i, (im, label) in test_df.iterrows():
#     new_path = f'words_data/test/{label.ljust(Params.MAX_LEN, "_")}'
#     if not os.path.exists(new_path):
#         os.makedirs(new_path)
#     im = Image.open(im)
#     im.save(f'{new_path}/{i}.png')
#     if i % (len(test_df) // 100 + 1) == 0:
#         print('.', end='')

# %% [markdown]
# # Datasets

# %%
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
        self.classes = [''.join([i if i != '_' else '' for i in word]) for word in self.classes]
        self.label_transform = label_transofrm


    def __getitem__(self, index):
        image, label = super(HandWritingDataset, self).__getitem__(index)
        if self.label_transform is not None:
            label = tensor(self.label_transform(self.classes[label]))
        return image, label

# %%
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
            labels = h5f['labels'][self.current_epoch * self.epoch_size : (self.current_epoch + 1) * self.epoch_size]
            df['length'] = (labels != 0).sum(axis=1)

        length_df = df['length'].value_counts().sort_index()
        length_dict = length_df.to_dict()

        df['class_length'] = df['length'].map(length_dict)
        df['result'] = 1.0 / df['class_length'] + epsilon

        return WeightedRandomSampler(df['result'].values, len(df['result']))


testing_dataset = H5Dataset('train_data.h5', 1)
train_sampler = testing_dataset.create_h5_sampler(0)

# %%
def create_h5_dataset(size: int=10):
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

    h5_dataset = HandWritingDataset(root='words_data/train', transform=h5_augument, label_transofrm=Params.encode_string)

    start_time = time.time()

    with h5py.File('train_data.h5', 'a') as h5f:
        h5f.create_dataset('images', shape=(size * len(h5_dataset), 1, 64, 128), dtype=np.uint8)
        h5f.create_dataset('labels', shape=(size * len(h5_dataset), Params.MAX_LEN), dtype=np.uint8)

        for i in range(len(h5_dataset)):
            for j in range(size):
                image, label = h5_dataset[i]
                image = image.numpy() * 255
                label = np.pad(label.numpy(), (0, Params.MAX_LEN - len(label)), 'constant', constant_values=0)
                h5f['images'][i * size + j] = image
                h5f['labels'][i * size + j] = label
                if all(label == 0):
                    print(label)
        
            remaining_time = (time.time() - start_time) * (len(h5_dataset) * size - i * size - j) / (i * size + j + 1)
            print(f'\r {i * size + j} / {len(h5_dataset) * size}    {int(remaining_time / 60)}:{int(remaining_time  % 60)}  ', end='')

# create_h5_dataset(1)

# %%
def collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images, dim=0)  # Stack images into a single tensor
    lengths = torch.tensor([len(label) for label in labels])  # Original lengths of labels
    labels = pad_sequence(labels, batch_first=True, padding_value=0)  # Pad labels
    return images, labels

# %%
def create_sampler(dataset: datasets.ImageFolder, epsilon: float=0.004):
    """
    epsilon: how strongly flatten the distribution? 0 means flat, the greater the more similar to original distribution, after 0.01 there is little difference
    """
    df = pd.DataFrame(dataset.samples, columns=['filename', 'class'])
    df['label'] = df['class'].apply(lambda x: ''.join([i for i in dataset.classes[x] if i != ' ']))
    df['length'] = df['label'].apply(len)
    length_df = df['length'].value_counts().sort_index()
    class_count = df['class'].value_counts().sort_index()
    df['class_length'] = df['length'].apply(lambda x: length_df.iloc[x-1])
    df['class_count'] = df['class'].apply(lambda x: class_count.iloc[x])
    df['result'] = 1.0 / df['class_length'] + epsilon
    return WeightedRandomSampler(df['result'].values, len(df['result']))

    # class_counts = np.bincount([label for _, label in dataset.samples])
    # class_weights = 1.0 / class_counts
    # sample_weights = [class_weights[label] for _, label in dataset.samples]
    # return WeightedRandomSampler(sample_weights, len(sample_weights))

# %% [markdown]
# ### sampler tests

# %%
# augument = transforms.Compose([
#     transforms.RandomRotation(15, expand=True, fill=(255,)),
#     transforms.RandomAffine(0, translate=(0.1, 0.1), fill=(255,)),
#     transforms.RandomPerspective(distortion_scale=0.5, p=0.5, fill=(255,)),
#     transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
#     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
# ])

# transform = transforms.Compose([
#     transforms.Grayscale(num_output_channels=1),
#     transforms.Resize((32, 64)),
#     transforms.ToTensor(),
#     transforms.Lambda(lambda x: 1.0 - x),
#     transforms.Normalize((0.5,), (0.5,)),
# ])

# train_dataset = HandWritingDataset(root='words_data/train', transform=transform, label_transofrm=Params.encode_string)
# val_dataset = HandWritingDataset(root='words_data/val', transform=transform, label_transofrm=Params.encode_string)
# test_dataset = HandWritingDataset(root='words_data/test', transform=transform, label_transofrm=Params.encode_string)

# %%
# dist_loader = DataLoader(
#     train_dataset,
#     batch_size=32,
#     num_workers=2,
#     shuffle=True,
#     collate_fn=collate_fn,
# )

# results = []
# for i, (ims, labels) in enumerate(dist_loader):
#     results.extend([Params.decode_string(i) for i in labels.tolist()])

# res_df = pd.DataFrame(results, columns=['word'])

# res_df.value_counts().plot(figsize=(20, 5), logy=True)
# plt.show()

# res_df['word'].apply(len).value_counts().sort_index().plot()
# plt.show()

# %%
sampler_dataset = H5Dataset('small.h5', 20)

dist_loader = DataLoader(
    sampler_dataset,
    batch_size=32,
    num_workers=2,
    sampler=sampler_dataset.create_h5_sampler(0),
    collate_fn=collate_fn,
)



repeat = 1
results = []
for waiting in range(repeat):
    for i, (ims, labels) in enumerate(dist_loader):
        results.extend([Params.decode_string(i) for i in labels.tolist()])

res_df = pd.DataFrame(results, columns=['word'])

(res_df.value_counts() / repeat).plot(figsize=(20, 5), logy=True)
plt.show()

res_df = res_df['word'].apply(len).value_counts().sort_index() / repeat
res_df.plot(ylim=(0, res_df.max() * 1.1), figsize=(20, 5))
plt.show()

# %% [markdown]
# # Train test predict

# %% [markdown]
# ### predict

# %%
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

# %% [markdown]
# ### train

# %%
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

            progress_bar.update(epoch + 1, batch_idx, train_loss / (batch_idx + 1))
        if scheduler is not None:
            scheduler.step()
        
        train_loss /= len(train_loader)
        val_loss = validator.validate()

        progress_bar.update(epoch + 1, len(train_loader) - 1, train_loss, val_loss)

        history.add_epoch(train_loss, val_loss)

        if early_stopping:
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

# %% [markdown]
# # Model

# %%
class TLModel(nn.Module):
    def __init__(self, hidden_size=512):
        super(TLModel, self).__init__()
        self.cnn = models.mobilenet_v2(pretrained=True).features[:7]

        for param in self.cnn.parameters():
            param.requires_grad = False
            
        self.gru = nn.GRU(input_size=8 * self.cnn[-1].out_channels, hidden_size=hidden_size, num_layers=1, batch_first=True, bidirectional=True, dropout=0.5)
        
        self.dense = nn.Linear(self.gru.hidden_size * (self.gru.bidirectional + 1), 27)

    def forward(self, x):
        x = self.cnn(x)

        x = x.permute(0, 3, 2, 1)
        x = x.flatten(2)

        x, _ = self.gru(x)

        x = self.dense(x)
        x = x.permute(1, 0, 2)
        x = F.log_softmax(x, dim=2)
        
        return x

# %% [markdown]
# # Training

# %% [markdown]
# ### prep

# %%
augument = transforms.Compose([
    transforms.RandomRotation(15, expand=True, fill=(255,)),
    transforms.RandomAffine(0, translate=(0.05, 0.05), fill=(255,)),
    transforms.RandomPerspective(distortion_scale=0.5, p=0.5, fill=(255,)),
    transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
])

transform = transforms.Compose([
    # transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 128)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: 1.0 - x),
    transforms.Normalize((0.5,), (0.5,)),
])

# %%
# train_dataset = HandWritingDataset(root='words_data/train', transform=transform, label_transofrm=Params.encode_string, augument=augument)
# val_dataset = HandWritingDataset(root='words_data/val', transform=transform, label_transofrm=Params.encode_string)
# test_dataset = HandWritingDataset(root='words_data/test', transform=transform, label_transofrm=Params.encode_string)

# %%
train_dataset = H5Dataset('dataset.h5', 100)
val_dataset = HandWritingDataset(root='words_data/val', transform=transform, label_transofrm=Params.encode_string)

# %%
train_loader = DataLoader(train_dataset, batch_size=64, num_workers=4, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=64, num_workers=4, shuffle=True, collate_fn=collate_fn)

# train_loader = DataLoader(val_dataset, batch_size=64, num_workers=4, collate_fn=collate_fn, sampler=create_sampler(val_dataset))
# val_loader = DataLoader(test_dataset, batch_size=64, num_workers=4, collate_fn=collate_fn)

# %%
model = TLModel(hidden_size=1024).to(device)
criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
early_stopping = EarlyStopping(patience=5)
history = History()

# %% [markdown]
# ### Train

# %%
for i in range(1):
    train_dataset = H5Dataset('dataset.h5', 1)
    # train_dataset.current_epoch = i % 2
    train_loader = DataLoader(train_dataset, batch_size=64, num_workers=4, sampler=train_dataset.create_h5_sampler(0), collate_fn=collate_fn)
    train(model, train_loader, val_loader, optimizer, criterion, epochs=1, early_stopping=early_stopping, history=history)

# %%
for param in model.cnn.parameters():
    param.requires_grad = True
optimizer = optim.Adam(model.parameters(), lr=1e-4)
for i in range(1):
    train_dataset = H5Dataset('dataset.h5', 1)
    # train_dataset.current_epoch = 1
    train_loader = DataLoader(train_dataset, batch_size=64, num_workers=4, sampler=train_dataset.create_h5_sampler(0), collate_fn = collate_fn)
    train(model, train_loader, val_loader, optimizer, criterion, epochs=1, early_stopping=early_stopping, history=history)

raise Exception("This is a custom exception message.")

# %%
predict_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4, collate_fn=collate_fn)
predictions = predict(model, predict_loader, 100)
for pred, label in predictions:
    print(f"{pred}  -  {label}")

# %% [markdown]
# # Tuning

# %%
# params = {
#     "hidden": [2048, 1024, 512],
#     "batch": [16, 32, 64],
#     "step": [1e-3, 1e-4],
# }
# start_time = time.time()
# results = []
# for i, (hidden, batch, step) in enumerate(product(*params.values())):
#     print(f"{i}. hidden: {hidden}, batch: {batch}, step: {step}")
#     model = TLModel(hidden_size=hidden).to(device)
#     optimizer = optim.Adam(model.parameters(), lr=step)
#     scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
#     criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
#     train_loader = DataLoader(train_dataset, batch_size=batch, num_workers=4, shuffle=True, collate_fn=collate_fn)
#     val_loader = DataLoader(val_dataset, batch_size=batch, num_workers=4, shuffle=True, collate_fn=collate_fn)

#     early_stopping = EarlyStopping(patience=5)
#     history = History()

#     train(model, train_loader, val_loader, optimizer, criterion, epochs=1, early_stopping=early_stopping, history=history)
#     results.append({
#         "hidden": hidden,
#         "batch": batch,
#         "step": step,
#         "train_loss": history.history['train_loss'][-1],
#         "val_loss": history.history['val_loss'][-1],
#     })
#     scheduler.step()
#     print(f"Time elapsed: {time.time() - start_time:.2f} seconds")

# results_df = pd.DataFrame(results)
# results_df = results_df.sort_values(by=['val_loss', 'train_loss'], ascending=[True, True])
# results_df.to_csv('results.csv', index=False)

# %%
# results_df['name'] = results_df.apply(
#     lambda x: f"{x['hidden']}   {x['batch']}   {x['step']}", axis=1)
# results_df.plot(x='name', 
#                 y=['train_loss', 'val_loss'], 
#                 kind='bar', 
#                 figsize=(18, 6), 
#                 title='Train and Validation Loss by Parameters')
# plt.ylabel('Loss')
# plt.xlabel('Parameters')
# plt.xticks(rotation=45, ha='right')
# plt.tight_layout()
# plt.show()

# %% [markdown]
# # Visual

# %%
visual_dataset = HandWritingDataset(root='words_data/train', transform=transform, label_transofrm=Params.encode_string, augument=augument)
visual_loader = DataLoader(visual_dataset, batch_size=16, num_workers=4, sampler=create_sampler(visual_dataset), collate_fn=collate_fn)

images, labels = next(iter(visual_loader))

fig, axes = plt.subplots(4, 4, figsize=(12, 8))
for i, ax in enumerate(axes.flatten()):
    if i >= len(images):
        break
    ax.imshow(images[i][0].cpu(), cmap='gray', aspect=1)
    ax.set_title(Params.decode_string(labels[i].tolist()))
    ax.axis('off')
plt.tight_layout()
plt.show()



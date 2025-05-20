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


# Get list of images and labels
with open('iam_words/words.txt', 'r') as f:
    words = f.readlines()
words = [word.strip() for word in words]
words = words[18:-1]
words = [w for w in words if ' err ' not in w]
words = [[w.split(' ')[0], w.split(' ')[-1]] for w in words]
words = [
    [f'iam_words/words/{w.split('-')[0]}/{w.split('-')[0]}-{w.split('-')[1]}/{w}.png', y] for w, y in words]
df = pd.DataFrame(words, columns=['filename', 'word'])
df = df[df['filename'].apply(os.path.exists)]


# Filter out invalid images
valid_rows = []
for i, (path, label) in df.iterrows():
    try:
        with Image.open(path) as img:
            img.verify()  # Verify that the file is a valid image
        valid_rows.append((path, label))
    except Exception as e:
        print(f"Skipping file {path} due to error: {e}")
    if i % (len(df) //100 + 1) == 0:
        print('.', end='')
print('\n')

df = pd.DataFrame(valid_rows, columns=df.columns)
df = df[df['word'].apply(lambda x: all(char in Params.vocab[1:]
                         for char in x) and len(x) <= Params.MAX_LEN)]


# Split data into sets convinient for torch dataset
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
test_df, val_df = train_test_split(test_df, test_size=0.5, random_state=42)
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)


# Save images in convinient for torch way
shutil.rmtree('words_data', ignore_errors=True)

for i, (im, label) in train_df.iterrows():
    new_path = f'words_data/train/{label.ljust(Params.MAX_LEN, "_")}'
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    im = Image.open(im)
    im.save(f'{new_path}/{i}.png')
    if i % (len(train_df) // 100 + 1) == 0:
        print('.', end='')
print()

for i, (im, label) in val_df.iterrows():
    new_path = f'words_data/val/{label.ljust(Params.MAX_LEN, "_")}'
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    im = Image.open(im)
    im.save(f'{new_path}/{i}.png')
    if i % (len(val_df) // 100 + 1) == 0:
        print('.', end='')
print()


for i, (im, label) in test_df.iterrows():
    new_path = f'words_data/test/{label.ljust(Params.MAX_LEN, "_")}'
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    im = Image.open(im)
    im.save(f'{new_path}/{i}.png')
    if i % (len(test_df) // 100 + 1) == 0:
        print('.', end='')

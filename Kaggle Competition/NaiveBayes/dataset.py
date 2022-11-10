import os
from random import randint

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

DATA_DIR = '../data/'

if os.path.exists(os.path.join(DATA_DIR, 'train.csv')):
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'), index_col='id')
else:
    train_df = pd.read_csv("https://raw.githubusercontent.com/OSU-AIClub/Fall-2022/main/Kaggle%20Competition/data/train.csv")
train_df

# Get list of input and output values for each training sample
X = train_df['review'].values
y = train_df['sentiment'].values
n_classes = np.unique(y)

# Randomly split into a training and validation datasets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=randint(0,100))

# Read test dataset from file or internet
if os.path.exists(os.path.join(DATA_DIR, 'test.csv')):
    test_df = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'), index_col='id')
else:
    test_df = pd.read_csv("https://raw.githubusercontent.com/OSU-AIClub/Fall-2022/main/Kaggle%20Competition/data/test.csv")
X_test = test_df['review'].values

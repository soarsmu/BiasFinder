import os
import pandas as pd

import torch

def read_imdb_data(fpath):
    data = pd.read_csv(fpath, header=None, sep="\t")
    return data

def read_twitter_data(fpath):
    data = pd.read_csv(fpath, header=None, sep="\t")
    return data

def read_imdb_train(data_dir):
    fpath = os.path.join(data_dir, "train.csv")
    df = read_imdb_data(fpath)
    return df[0].values, df[1].values

def read_imdb_test(data_dir):
    fpath = os.path.join(data_dir, "test.csv")
    df = read_imdb_data(fpath)
    return df[0].values, df[1].values

def read_twitter_train(data_dir):
    fpath = os.path.join(data_dir, "train.csv")
    df = read_twitter_data(fpath)
    return df[0].values, df[1].values

def read_twitter_test(data_dir):
    fpath = os.path.join(data_dir, "test.csv")
    df = read_twitter_data(fpath)
    return df[0].values, df[1].values


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

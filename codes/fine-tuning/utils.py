import os
import pandas as pd

import torch

def read_csv_data(fpath):
    data = pd.read_csv(fpath, header=None, sep="\t")
    return data

def read_imdb_train(data_dir):
    fpath = os.path.join(data_dir, "train.csv")
    df = read_csv_data(fpath)
    return df[0].values, df[1].values

def read_imdb_test(data_dir):
    fpath = os.path.join(data_dir, "test.csv")
    df = read_csv_data(fpath)
    return df[0].values, df[1].values

def read_twitter_train(data_dir):
    fpath = os.path.join(data_dir, "train.csv")
    df = read_csv_data(fpath)
    return df[0].values, df[1].values

def read_twitter_test(data_dir):
    fpath = os.path.join(data_dir, "test.csv")
    df = read_csv_data(fpath)
    return df[0].values, df[1].values

def read_train_data(data_dir):
    fpath = os.path.join(data_dir, "train.csv")
    df = read_csv_data(fpath)
    return df[0].values, df[1].values

def read_test_data(data_dir):
    fpath = os.path.join(data_dir, "test.csv")
    df = read_csv_data(fpath)
    return df[0].values, df[1].values

def read_original_data(data_dir) :
    fpath = os.path.join(data_dir, "original.csv")
    df = read_csv_data(fpath)
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

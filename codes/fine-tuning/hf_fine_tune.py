import os
import pandas as pd

from sklearn.model_selection import train_test_split
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertForSequenceClassification, Trainer, TrainingArguments


def read_imdb_data(fpath) :
    data = pd.read_csv(fpath, header=None, sep="\t")
    return data


def read_imdb_train(data_dir):
    fpath = os.path.join(data_dir, "train.csv")
    df = read_imdb_data(fpath)
    return df[1].values, df[0].values


def read_imdb_test(data_dir):
    fpath = os.path.join(data_dir, "test.csv")
    df = read_imdb_data(fpath)
    return df[1].values, df[0].values

class IMDbDataset(torch.utils.data.Dataset):
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

def check_data():
    print(len(train_texts))
    print(train_texts[:5])
    print(len(train_labels))
    print(train_labels[:5])


if __name__ == "__main__" :

    DATA_DIR = "./../../asset/imdb/"

    train_texts, train_labels = read_imdb_train(DATA_DIR)
    test_texts, test_labels = read_imdb_test(DATA_DIR)

    # check_data()

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=.1)

    train_texts = list(train_texts)
    val_texts = list(val_texts)
    test_texts = list(test_texts)
    train_labels = list(train_labels)
    val_labels = list(val_labels)
    test_labels = list(test_labels)

    model_name = "bert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # check_data()

    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True)


    train_dataset = IMDbDataset(train_encodings, train_labels)
    val_dataset = IMDbDataset(val_encodings, val_labels)
    test_dataset = IMDbDataset(test_encodings, test_labels)


    training_args = TrainingArguments(
        output_dir='./results',          # output directory
        num_train_epochs=3,              # total number of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=64,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=10,
    )

    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    trainer = Trainer(
        # the instantiated 🤗 Transformers model to be trained
        model=model,
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=val_dataset             # evaluation dataset
    )

    trainer.train()

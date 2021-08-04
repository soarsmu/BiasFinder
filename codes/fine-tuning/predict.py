import os, gc
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

import pickle
from tqdm import tqdm


import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import DataLoader

from utils import read_imdb_test, IMDbDataset


if __name__ == "__main__":

    # DATA_DIR = "./../../data/biasfinder/gender/"
    DATA_DIR = "./../../data/eec/gender/"

    test_texts, test_labels = read_imdb_test(DATA_DIR)

    # test_texts = list(test_texts)[:1000]
    # test_labels = list(test_labels)[:1000]

    test_texts = list(test_texts)
    test_labels = list(test_labels)


    model_name = "bert-base-uncased"
    # model_name = "bert-base-cased"
    # model_name = "roberta-base"
    # model_name = "microsoft/deberta-large-mnli"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    test_encodings = tokenizer(
        test_texts, truncation=True, padding=True, max_length=512)
    test_dataset = IMDbDataset(test_encodings, test_labels)

    
    if model_name == "bert-base-uncased" :
        checkpoint_name = "./results/bert-base-uncased/gpu1/checkpoint-2000"
    elif model_name == "bert-base-cased" :
        checkpoint_name = "./results/bert-base-cased/gpu0/checkpoint-2000"
    elif model_name == "roberta-base":
        checkpoint_name = "./results/roberta-base/gpu1/checkpoint-4000"
        
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_name)

    # Define test trainer
    test_trainer = Trainer(model)

    # Make prediction
    # raw_pred, _, _ = test_trainer.predict(test_dataset)
    
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    raw_pred, _, _ = test_trainer.prediction_loop(
        test_loader, description="prediction")

    # Preprocess raw predictions
    y_pred = np.argmax(raw_pred, axis=1)


    fpath = os.path.join(DATA_DIR, "prediction.pkl")

    with open(fpath, 'wb') as f:
        pickle.dump(y_pred, f)



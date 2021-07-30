import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

from utils import read_imdb_test, IMDbDataset


def compute_metrics(pred, labels):
    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred)
    precision = precision_score(y_true=labels, y_pred=pred)
    f1 = f1_score(y_true=labels, y_pred=pred)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

if __name__ == "__main__":

    DATA_DIR = "./../../asset/imdb/"

    test_texts, test_labels = read_imdb_test(DATA_DIR)

    test_texts = list(test_texts)
    test_labels = list(test_labels)

    # model_name = "bert-base-cased"
    # model_name = "bert-base-uncased"
    model_name = "roberta-base"
    # model_name = "microsoft/deberta-large-mnli"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    test_encodings = tokenizer(
        test_texts, truncation=True, padding=True, max_length=512)
    test_dataset = IMDbDataset(test_encodings, test_labels)

    if model_name == "bert-base-uncased" :
        checkpoint_name = "./results/bert-base-uncased/gpu1/checkpoint-42000"
        # checkpoint_name = "./results/bert-base-uncased/gpu1/checkpoint-2500"
    elif model_name == "bert-base-cased" :
        checkpoint_name = "./results/bert-base-cased/gpu0/checkpoint-500"
    elif model_name == "roberta-base":
        checkpoint_name = "./results/roberta-base/gpu1/checkpoint-4000"
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_name)

    # Define test trainer
    test_trainer = Trainer(model)

    # Make prediction
    raw_pred, _, _ = test_trainer.predict(test_dataset)

    # Preprocess raw predictions
    y_pred = np.argmax(raw_pred, axis=1)

    metrics = compute_metrics(y_pred, test_labels)
    print(metrics)



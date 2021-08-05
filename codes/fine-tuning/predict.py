import os, gc
import pandas as pd
import numpy as np
import argparse
import pickle
from tqdm import tqdm


import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import DataLoader

from utils import read_imdb_test, read_twitter_test, BiasFinderDataset


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mutation-tool', default="biasfinder", type=str)
    parser.add_argument('--bias-type', default="gender", type=str)
    parser.add_argument('--task', default="imdb", type=str, help='dataset for fine-tuning the model')
    parser.add_argument('--model', default='bert-base-uncased')
    parser.add_argument('--mutant', default='imdb', help='dataset utilized to generate mutant')
    parser.add_argument('--batch-size', default=64, type=int)

    return parser.parse_args()

def predict():

    args = get_args()


    data_dir = f"./../../data/{args.mutation_tool}/{args.bias_type}/{args.mutant}/"


    # test_labels, test_texts  = read_imdb_test(data_dir)
    test_labels, test_texts = read_twitter_test(data_dir)

    # test_texts = list(test_texts)[:1000]
    # test_labels = list(test_labels)[:1000]

    test_texts = list(test_texts)
    test_labels = list(test_labels)


    model_name = args.model
    # model_name = "bert-base-uncased"
    # model_name = "bert-base-cased"
    # model_name = "roberta-base"
    # model_name = "microsoft/deberta-large-mnli"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    test_encodings = tokenizer(
        test_texts, truncation=True, padding=True, max_length=512)
    test_dataset = BiasFinderDataset(test_encodings, test_labels)

    base_checkpoint = f"./models/{args.task}/{args.model}/"
    if args.task == "imdb" :
        if model_name == "bert-base-uncased" :
            checkpoint_name = base_checkpoint + "gpu1/checkpoint-2000"
            # checkpoint_name = "./models/twitter_semeval/bert-base-uncased/gpu1/checkpoint-500"
        elif model_name == "bert-base-cased" :
            checkpoint_name = "./results/bert-base-cased/gpu0/checkpoint-2000"
        elif model_name == "roberta-base":
            checkpoint_name = "./results/roberta-base/gpu1/checkpoint-4000"
    elif args.task == "twitter_semeval" :
        if model_name == "bert-base-uncased":
            checkpoint_name = base_checkpoint + "gpu1/checkpoint-500"

    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_name)

    # Define test trainer
    test_trainer = Trainer(model)

    # Make prediction
    # raw_pred, _, _ = test_trainer.predict(test_dataset)
    
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    raw_pred, _, _ = test_trainer.prediction_loop(
        test_loader, description="prediction")

    # Preprocess raw predictions
    y_pred = np.argmax(raw_pred, axis=1)


    fpath = os.path.join(data_dir, "prediction.pkl")

    with open(fpath, 'wb') as f:
        pickle.dump(y_pred, f)


if __name__ == "__main__":
    predict()

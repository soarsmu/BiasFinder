import gc
from utils import read_imdb_test, read_twitter_test, CustomDataset
from torch.utils.data import DataLoader
from pathlib import Path
import json
from tqdm import tqdm
import pickle
import argparse
import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments



def compute_metrics(pred, labels):
    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred)
    precision = precision_score(y_true=labels, y_pred=pred)
    f1 = f1_score(y_true=labels, y_pred=pred)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default="imdb", type=str,
                        help='dataset for fine-tuning the model')
    parser.add_argument('--model', default='bert-base-uncased')
    parser.add_argument('--dataset', default='imdb',
                        help='dataset for evaluating the performance of model')
    parser.add_argument('--batch-size', default=64, type=int)

    return parser.parse_args()


def find_best_checkpoint(checkpoint_dir):
    """
    Find the best checkpoint in the directory
    :param checkpoint_dir: dir where the checkpoints are being saved.
    :return: the checkpoint where it achieve lowest loss
    """
    ckpt_files = os.listdir(checkpoint_dir)  # list of strings
    steps = [int(filename[11:])
             for filename in ckpt_files]  # 'epoch={int}.ckpt' filename format
    # print(steps)
    last_step = max(steps)
    training_state_fpath = os.path.join(
        checkpoint_dir, f"checkpoint-{last_step}/trainer_state.json")

    f = open(training_state_fpath,)  # Opening JSON file
    data = json.load(f)  # returns JSON object as a dictionary

    return data["best_model_checkpoint"]


def test():
    print("=== Test ===")

    args = get_args()
    print(args)


    data_dir = f"./../../asset/{args.dataset}/"

    if args.dataset == "imdb":
        test_labels, test_texts = read_imdb_test(data_dir)
    elif args.dataset == "twitter_semeval" or args.dataset == "twitter_s140":
        test_labels, test_texts = read_twitter_test(data_dir)
    else:
        raise ValueError("Unknown dataset used for generating mutants")

    # test_texts = list(test_texts)[:100]
    # test_labels = list(test_labels)[:100]

    test_texts = list(test_texts)
    test_labels = list(test_labels)

    model_name = args.model
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    test_encodings = tokenizer(
        test_texts, truncation=True, padding=True, max_length=512)
    test_dataset = CustomDataset(test_encodings, test_labels)

    checkpoint_dir = f"./models/{args.task}/{args.model}/"
    best_checkpoint = find_best_checkpoint(checkpoint_dir)

    model = AutoModelForSequenceClassification.from_pretrained(best_checkpoint)

    test_trainer = Trainer(model)

    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False)
    raw_pred, _, _ = test_trainer.prediction_loop(
        test_loader, description="prediction")

    # Preprocess raw predictions
    y_pred = np.argmax(raw_pred, axis=1)

    metrics = compute_metrics(y_pred, test_labels)
    print(metrics)

    fpath = os.path.join(data_dir, f"predictions/{args.model}.pkl")

    parent_dir = "/".join(str(fpath).split('/')[:-1])
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    with open(fpath, 'wb') as f:
        pickle.dump(y_pred, f)


if __name__ == "__main__":
    test()

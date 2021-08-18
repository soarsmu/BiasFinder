###
# source
# https://towardsdatascience.com/fine-tuning-pretrained-nlp-models-with-huggingfaces-trainer-6326a4456e7b
#

import os
import pandas as pd
import numpy as np
import argparse
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import EarlyStoppingCallback


from utils import read_imdb_train, read_twitter_train, CustomDataset


def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred)
    precision = precision_score(y_true=labels, y_pred=pred)
    f1 = f1_score(y_true=labels, y_pred=pred)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='bert-base-uncased')
    parser.add_argument('--task', default="imdb", type=str)
    parser.add_argument('--test-size', default=0.4, type=float)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--train-bs', default=8, type=int)
    parser.add_argument('--learning-rate', default=2e-5, type=float)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--warmup-steps', default=500, type=int)
    parser.add_argument('--logging-steps', default=500, type=int)
    parser.add_argument('--eval-steps', default=500, type=int)
    parser.add_argument('--save-steps', default=500, type=int)
    parser.add_argument('--weight-decay', default=0.01, type=float)

    return parser.parse_args()
    

def fine_tune() :

    print("=== Fine-tune ===")

    args = get_args()
    print(args)
    
    if args.task ==  "imdb" :
        data_dir = "./../../asset/imdb/"
        train_labels, train_texts = read_imdb_train(data_dir)
    elif args.task == "twitter_semeval":
        data_dir = "./../../asset/twitter_semeval/"
        train_labels, train_texts = read_twitter_train(data_dir)
    elif args.task == "twitter_s140":
        data_dir = "./../../asset/twitter_s140/"
        train_labels, train_texts = read_twitter_train(data_dir)

    # check_data()

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=args.test_size)
    
    ## IF HAVE MUCH TIME, try to increase test size because the fine-tuning run fast

    train_texts = list(train_texts)
    val_texts = list(val_texts)
    train_labels = list(train_labels)
    val_labels = list(val_labels)

    model_name = args.model
    # model_name = "bert-base-cased"
    # model_name = "roberta-base"
    # model_name = "microsoft/deberta-large-mnli"
    # model_name = "bert-base-uncased"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # check_data()

    train_encodings = tokenizer(
        train_texts, truncation=True, padding=True, max_length=512)
    val_encodings = tokenizer(
        val_texts, truncation=True, padding=True, max_length=512)    

    train_dataset = CustomDataset(train_encodings, train_labels)
    val_dataset = CustomDataset(val_encodings, val_labels)

    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    training_args = TrainingArguments(
        # output directory
        output_dir=f'./models/{args.task}/{model_name}/',
        num_train_epochs=args.epochs,              # total number of training epochs
        per_device_train_batch_size=args.train_bs,  # batch size per device during training
        per_device_eval_batch_size=64,   # batch size for evaluation
        warmup_steps=args.warmup_steps,                # number of warmup steps for learning rate scheduler
        weight_decay=args.weight_decay,               # strength of weight decay
        # directory for storing logs
        logging_dir=f'./logs/{args.task}/{model_name}/',
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        seed=0,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps, 
        save_total_limit=5,
        save_steps=args.save_steps,
        load_best_model_at_end=True
    )

    # trainer = Trainer(
    #     # the instantiated 🤗 Transformers model to be trained
    #     model=model,
    #     args=training_args,                  # training arguments, defined above
    #     train_dataset=train_dataset,         # training dataset
    #     eval_dataset=val_dataset,             # evaluation dataset
    #     compute_metrics=compute_metrics,
    # )

    trainer = Trainer(
        # the instantiated 🤗 Transformers model to be trained
        model=model,
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=val_dataset,             # evaluation dataset
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=7)],

    )

    trainer.train()


if __name__ == "__main__" :
    fine_tune()

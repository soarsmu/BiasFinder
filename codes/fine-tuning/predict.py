import os, gc
import pandas as pd
import numpy as np
import argparse
import pickle
from tqdm import tqdm
import json
from pathlib import Path
import sys

import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import DataLoader

from utils import read_imdb_test, read_twitter_test, read_test_data, read_original_data, CustomDataset


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mutation-tool', default="biasfinder", type=str)
    parser.add_argument('--bias-type', default="gender", type=str)
    parser.add_argument('--task', default="imdb", type=str, help='dataset for fine-tuning the model')
    parser.add_argument('--model', default='bert-base-uncased')
    parser.add_argument('--mutant', default='imdb', help='dataset utilized to generate mutant')
    parser.add_argument('--type', default='mutant', help='mutant | original')

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
    training_state_fpath = os.path.join(checkpoint_dir, f"checkpoint-{last_step}/trainer_state.json")

    f = open(training_state_fpath,)  # Opening JSON file
    data = json.load(f)  # returns JSON object as a dictionary
    
    return data["best_model_checkpoint"]

def generate_original_data(data_dir, mutation_tool):
    if mutation_tool == "biasfinder" :
        headers = ["label", "mutant", "template", "original", "gender"]
    elif mutation_tool == "eec" :
        headers = ["label", "mutant", "template",
                   "original", "person", "gender", "emotion"]
    elif mutation_tool ==  "mtnlp" :
        headers = ["label", "mutant", "original", "mutation_type"]
    df = pd.read_csv(data_dir + "test.csv", header=None, sep="\t",
                     names=headers)
    ori_df = df[["label", "original"]]
    ori_df = ori_df.drop_duplicates()
    ori_df.to_csv(data_dir + "original.csv",
                   index=None, header=None, sep="\t")

def batch_tokenizer(tokenizer, test_texts, batch_size=5000):

    n = len(test_texts)
    i = 0 
    lb = i * batch_size # lower bound
    ub = (i+1) * batch_size # upper bound
    
    input_ids = None
    token_type_ids = None
    attention_mask = None
    while lb < n :
        curr_test_encodings = tokenizer(test_texts[lb:ub], truncation=True, padding=True, max_length=512)
        if str(type(input_ids)) == "<class 'numpy.ndarray'>":
            input_ids = np.concatenate((input_ids, curr_test_encodings['input_ids']), axis=0)
            token_type_ids = np.concatenate((token_type_ids, curr_test_encodings['token_type_ids']), axis=0)
            attention_mask = np.concatenate((attention_mask, curr_test_encodings['attention_mask']), axis=0)
        else :
            input_ids = np.array(curr_test_encodings['input_ids'])
            token_type_ids = np.array(curr_test_encodings['token_type_ids'])
            attention_mask = np.array(curr_test_encodings['attention_mask'])
            
        i += 1
        lb = i * batch_size  # lower bound
        ub = (i+1) * batch_size  # upper bound

        if ub > n :
            ub = n
    
    return {'input_ids': input_ids, 
            'token_type_ids': token_type_ids, 
            'attention_mask': attention_mask}


def predict():
    print("=== Predict ===")
    args = get_args()
    print(args)

    if args.bias_type !=  "" :
        data_dir = f"./../../data/{args.mutation_tool}/{args.bias_type}/{args.mutant}/"
    else :
        data_dir = f"./../../data/{args.mutation_tool}/{args.mutant}/"

    if args.type == "mutant" :
        test_labels, test_texts = read_test_data(data_dir)
        # if args.mutant == "imdb" :
        #     test_labels, test_texts  = read_imdb_test(data_dir)
        # elif args.mutant == "twitter_semeval" or args.mutant == "twitter_s140" :
        #     test_labels, test_texts = read_twitter_test(data_dir)
        # else :
        #     raise ValueError("Unknown dataset used for generating mutants")
    elif args.type == "original" :
        # if not os.path.exists(data_dir + "original.csv") :
        generate_original_data(data_dir, mutation_tool=args.mutation_tool)
        test_labels, test_texts = read_original_data(data_dir)
    else:
        raise ValueError("Unknown type that needs to be tested")

    # test_texts = list(test_texts)[:100]
    # test_labels = list(test_labels)[:100]

    test_texts = list(test_texts)
    test_labels = list(test_labels)


    model_name = args.model
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    test_encodings = batch_tokenizer(tokenizer, test_texts, batch_size=10000)

    # print(len(test_encodings))
    # print(len(test_labels))
    # assert len(test_encodings) == len(test_labels)
    test_dataset = CustomDataset(test_encodings, test_labels)

    print("passed")

    checkpoint_dir = f"./models/{args.task}/{args.model}/"
    best_checkpoint = find_best_checkpoint(checkpoint_dir)

    model = AutoModelForSequenceClassification.from_pretrained(best_checkpoint)

    test_trainer = Trainer(model)
    
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    raw_pred, _, _ = test_trainer.prediction_loop(
        test_loader, description="prediction")

    # Preprocess raw predictions
    y_pred = np.argmax(raw_pred, axis=1)

    fpath = os.path.join(data_dir, f"{args.type}-predictions/{args.model}.pkl")
    
    parent_dir = "/".join(str(fpath).split('/')[:-1])
    if not os.path.exists(parent_dir) :
        os.makedirs(parent_dir)

    with open(fpath, 'wb') as f:
        pickle.dump(y_pred, f)


if __name__ == "__main__":
    predict()

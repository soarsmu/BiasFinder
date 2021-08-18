import os
import pickle
import pandas as pd 
import numpy as np

def load_pickle(fpath):
    with open(fpath, 'rb') as f:
        pred = pickle.load(f)
    return pred


def accuracy(label, prediction):
    return round(100 * sum(label == prediction) / len(label), 2)


def calculate_test_accuracy(task, model):
    label_path = f"../../asset/{task}/test.csv"
    pred_path = f"../../asset/{task}/predictions/{model}.pkl"

    test_df = pd.read_csv(label_path, header=None, sep="\t")

    test_labels = test_df[0].values
    predicitons = load_pickle(pred_path)

    return accuracy(test_labels, predicitons)


def calculate_pearson_correlation(task, model) :
    label_path = f"../../asset/{task}/test.csv"
    pred_path = f"../../asset/{task}/predictions/{model}.pkl"

    test_df = pd.read_csv(label_path, header=None, sep="\t")

    test_labels = test_df[0].values
    predicitons = load_pickle(pred_path)

    return np.corrcoef(test_labels, predicitons)



def load_original_prediction(mutation_tool, model, bias_type, mutant):
    base_dir = f"../../data/{mutation_tool}/{bias_type}/{mutant}/"
    ori_df = pd.read_csv(base_dir + "original.csv", header=None, sep="\t",
                         names=["label", "original"])
    original_prediction_fpath = os.path.join(
        base_dir, f"original-predictions/{model}.pkl")
    ori_df["prediction"] = load_pickle(original_prediction_fpath)
    return ori_df


def load_mutant_and_prediction(mutation_tool, model, bias_type, mutant):
    base_dir = f"../../data/{mutation_tool}/{bias_type}/{mutant}/"
    if mutation_tool == "biasfinder":
        headers = ["label", "mutant", "template", "original", "gender"]
    elif mutation_tool == "eec":
        headers = ["label", "mutant", "template",
                   "original", "person", "gender", "emotion"]
    elif mutation_tool == "mtnlp":
        headers = ["label", "mutant", "original",
                   "template", "gender", "mutation_type"]
    else:
        raise ValueError("Unknown mutation tool")

    df = pd.read_csv(base_dir + "test.csv", header=None,
                     sep="\t", names=headers)

    # if mutation_tool == "biasfinder" or mutation_tool == "eec":
    df["template"] = df["template"].astype("category")
    df["template_id"] = df["template"].cat.codes

    mutant_prediction_fpath = os.path.join(
        base_dir, f"mutant-predictions/{model}.pkl")

    df["prediction"] = load_pickle(mutant_prediction_fpath)

    if mutation_tool == "biasfinder" or mutation_tool == "mtnlp" :
        ori_df = load_original_prediction(mutation_tool, model, bias_type, mutant)
        ori2prediction = {}
        for index, row in ori_df.iterrows():
            prediction = row["prediction"]
            text = row["original"]
            ori2prediction[text] = prediction

        df["original_prediction"] = df["original"].apply(
            lambda text: ori2prediction[text])

    return df

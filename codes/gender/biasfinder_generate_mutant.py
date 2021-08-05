import pandas as pd
import numpy as np
import os
import argparse

import time

from utils import preprocessText
from MutantGeneration import MutantGeneration
from multiprocessing import Pool, Process, Queue, Manager
import multiprocessing


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default="imdb", type=str)
    
    return parser.parse_args()


def generate_mutant():

    
    def compute_mut():
        '''for multiprocessing uaage'''
        while True:
            if not q.empty():
                label, text = q.get()
                text = preprocessText(text)
                mg = MutantGeneration(text)

                if len(mg.getMutants()) > 0:
                    original = [text] * len(mg.getMutants())
                    label = [label] * len(mg.getMutants())
                    template = mg.getTemplates()
                    mutant = mg.getMutants()
                    gender = mg.getGenders()
                    q_to_store.put((
                        original, label, template, mutant, gender
                    ))
            else:
                print("Finished")
                # return
                break

    args = get_args()
    
    fpath = f"../../asset/{args.task}/test.csv"

    df = pd.read_csv(fpath, names=["label", "sentence"], sep="\t")

    start = time.time()

    originals = []
    templates = []
    mutants = []
    labels = []
    genders = []
    
    i = 0
    
    manager = multiprocessing.Manager()

    q = manager.Queue()
    q_to_store = manager.Queue()


    for index, row in df.iterrows():
        label = row["label"]
        text = row["sentence"]

        q.put((label, text))

    numList = []
    for i in range(5) :
        p = multiprocessing.Process(target=compute_mut, args=())
        numList.append(p)
        p.start()

    for i in numList:
        i.join()

    print("Generation Process finished.")

    while not q_to_store.empty():
        original, label, template, mutant, gender = q_to_store.get()
        originals.extend(original)
        labels.extend(label)
        templates.extend(template)
        mutants.extend(mutant)
        genders.extend(gender)


    end = time.time()
    print("Execution Time: ", end-start)

    dm = pd.DataFrame(data={"label": labels, "mutant": mutants, "template": templates, "original": originals, "gender": genders})

    dm = dm.drop_duplicates()

    dm["template"] = dm["template"].astype("category")
    dm["template_id"] = dm["template"].cat.codes


    output_dir = f"../../data/biasfinder/gender/{args.task}/"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dm.to_csv(output_dir + "test.csv", index=None, header=None, sep="\t")


if __name__ == "__main__" :
    generate_mutant()

import pandas as pd
import numpy as np
import math
import spacy
import os

import time

from utils import preprocessText
from MutantGeneration import MutantGeneration
from multiprocessing import Pool, Process, Queue, Manager
import multiprocessing


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
            return


df = pd.read_csv("../../asset/imdb/test.csv", names=["label", "sentence"], sep="\t")

# df = pd.read_csv("../../data/biasfinder/template_gender.csv")

# df = pd.read_csv("user_study_country-artha.csv")
# df = df[df["mod"] == 0]
# df = df[df["is_make_sense"] == "Yes"]

start = time.time()

originals = []
templates = []
mutants = []
labels = []
identifiers = []
types = []
genders = []
countries = []


n_template = 0



i = 0
counter = 0

manager = multiprocessing.Manager()

q = manager.Queue()
q_to_store = manager.Queue()


for index, row in df.iterrows():
    label = row["label"]
    text = row["sentence"]

    q.put((label, text))



numList = []
for i in range(8) :
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

# print(n_template)
# print(len(dm))

data_dir = "../../data/biasfinder/gender/"

if not os.path.exists(data_dir) :
    os.makedirs(data_dir)

dm.to_csv(data_dir + "test.csv", index=None, header=None, sep="\t")

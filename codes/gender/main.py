import pandas as pd
import numpy as np
import math
import spacy
import os

import time

from utils import preprocessText
from MutantGeneration import MutantGeneration

# df = pd.read_csv("../../data/imdb/test.csv", names=["label", "sentence"], sep="\t")

df = pd.read_csv("../../data/biasfinder_backup/gender-template.csv")

start = time.time()

originals = []
templates = []
mutants = []
labels = []
identifiers = []
types = []
genders = []
countries = []

i = 0
counter = 0
for index, row in df.iterrows():
    label = row["label"]
    text = row["sentence"]
    text = preprocessText(text)
    mg = MutantGeneration(text)
    i += 1
    if i%200 == 0 : print(i)
            
    if len(mg.getMutants()) > 0:    
        originals.extend([text] * len(mg.getMutants()))
        labels.extend([label] * len(mg.getMutants()))
        templates.extend(mg.getTemplates())
        mutants.extend(mg.getMutants())
        genders.extend(mg.getGenders())

end = time.time()
print("Execution Time: ", end-start)

dm = pd.DataFrame(data={"label": labels, "mutant": mutants, "template": templates, "original": originals, "gender": genders})
dm

dm["template"] = dm["template"].astype("category")
dm["template_id"] = dm["template"].cat.codes

data_dir = "../../data/biasfinder/gender/"

if not os.path.exists(data_dir) :
    os.makedirs(data_dir)

dm.to_csv(data_dir + "test.csv", index=None, header=None, sep="\t")
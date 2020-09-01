import pandas as pd
import numpy as np
import math
import spacy
import os

import time

from utils import preprocessText
from CountryMutantGeneration import CountryMutantGeneration

df = pd.read_csv("../../asset/imdb/test.csv", names=["label", "original"], sep="\t")

# df = pd.read_csv("../../data/biasfinder/template_gender.csv")

start = time.time()

originals = []
templates = []
mutants = []
labels = []
names = []
countries = []

i = 0
counter = 0
n_template = 0
for index, row in df[:100].iterrows():
    label = row["label"]
    text = row["original"]
    text = preprocessText(text)
    mg = CountryMutantGeneration(text)
    i += 1
    if i%20 == 0 : 
        print(i)
        print(n_template)
            
    if len(mg.getMutants()) > 0: 
        n_template += 1
        originals.extend([text] * len(mg.getMutants()))
        labels.extend([label] * len(mg.getMutants()))
        templates.extend(mg.getTemplates())
        mutants.extend(mg.getMutants())
        names.extend(mg.getNames())
        countries.extend(mg.getCountries())

end = time.time()
print("Execution Time: ", end-start)

dm = pd.DataFrame(data={"label": labels, "mutant": mutants, "template": templates, "original": originals, "names": names, "countries": countries})
dm

dm["template"] = dm["template"].astype("category")
dm["template_id"] = dm["template"].cat.codes

data_dir = "../../data/biasfinder/country/"

if not os.path.exists(data_dir) :
    os.makedirs(data_dir)

dm.to_csv(data_dir + "test.csv", index=None, header=None, sep="\t")
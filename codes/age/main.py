# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 20:05:50 2020

@author: MSI
"""


import spacy
import pandas as pd
import inflect


datasetLoc = "./asset/test.csv"
# gawLoc =  './asset/gaw.txt'
# occListLoc = "./asset/neutral-occupation.csv"

# occ = pd.read_csv(occListLoc)
# occList = occ['occupation'].to_list()

df = pd.read_csv(datasetLoc, header=None, sep='\t')
df.columns = ['sentiment', 'review']
# df = df.sample(n = 1000, random_state = 1234)

# gaw = pd.read_csv(gawLoc, sep=",", header=None)
# gaw.columns = ['x', 'y']
# gaw = gaw['x'].append(gaw['y']).reset_index(drop=True)
# gawList = gaw.to_list()
nlp = spacy.load('en_core_web_sm')
p = inflect.engine()

refLoc = "./asset/ref.csv"
refList = pd.read_csv(refLoc)
refList = refList['ref'].to_list()

 
from spacy.matcher import Matcher
matcher = Matcher(nlp.vocab)
  
def isValidAdj(spacySpan):
    if spacySpan.root.head.text in refList:
        return True
    else:
        return False

def addPattern(listOfAdj, matcher):
    for element in listOfAdj:
        pattern = [{"LOWER": element}]
        matcher.add(element, None, pattern)
    matcher.add(element, None, pattern)

def createTempl(spacySpan):
    spacyDoc = spacySpan.doc
    if spacySpan.text.lower() in ["younger", "older", "elder"]:
        placeholder = "@adjComparative"
    elif spacySpan.text.lower() in ["youngest", "oldest", "eldest"]:
        placeholder = "@adjSuperlative"
    else:
        placeholder = "@adj"
    
    template = ''
    if len(spacySpan) == 1:
        if spacySpan.root.i != 0:      
            ending = spacyDoc[spacySpan.end:]
            if checkDeterminer(spacySpan):
                beginning = spacyDoc[0:spacySpan.start-1]    
                template = beginning.text + " {} {} ".format("@det", placeholder) + ending.text
            else:
                beginning = spacyDoc[0:spacySpan.start]
                template = beginning.text + " {} ".format(placeholder) + ending.text
        else:
            ending = spacyDoc[spacySpan.end:]
            template = "{} ".format(placeholder) + ending.text
    # else:
    #     if spacySpan.root.i != 0:
    #         beginning = spacyDoc[0:spacySpan.start]
    #         ending = spacyDoc[spacySpan.end:]
    #         template = beginning.text + " {} ".format("@ageNumber") + ending.text
    #     else:
    #         ending = spacyDoc[spacySpan.end:]
    #         template = "{} ".format(placeholder) + ending.text
    return template
    
def searchAdj(spacyDoc):
    match = ()
    validSpan = None
    match = matcher(spacyDoc)
    if len(match) > 0:
        for match_id, start, end in match:
            span = spacyDoc[start:end]
            if isValidAdj(span):
                validSpan = span
                break
        return validSpan
    else:
        return validSpan

def checkDeterminer(span):
    if span.root.i != 0:
        if span.root.nbor(-1).text.lower() in ['a', 'an']:
            return True
    return False

def pipelineGenerateTempl(spacyDoc):
    template = ''
    adjSpan = searchAdj(spacyDoc)
    if adjSpan != None:
        template = createTempl(adjSpan)
    return template

import time
start_time = time.time()

listOfAdj = ['old', 'young', 'older', 'younger', 'oldest', 'youngest', 'elder', 'eldest']
listOfPlaceholder = ['old', 'young', 'older', 'younger', 'oldest', 'youngest', '25 years old', '65 years old']
addPattern(listOfAdj, matcher)  
templateList = []
counter = 0
for index, row in df.iterrows():
    template = ''
    print("counter: {}".format(counter))
    text = row.review
    doc = nlp(text)
    template = pipelineGenerateTempl(doc)
    if '@adjComparative' in template:
        oldMutant = template.replace("@adjComparative", "older").replace("@det", "an")
        youngMutant = template.replace("@adjComparative", "younger").replace("@det", "a")
        templateList.append((row.sentiment, oldMutant, oldMutant, template, "old", text))
        templateList.append((row.sentiment, youngMutant, youngMutant, template, "young", text))
    
    elif '@adjSuperlative' in template:
        oldMutant = template.replace("@adjSuperlative", "older").replace("@det", "an")
        youngMutant = template.replace("@adjSuperlative", "younger").replace("@det", "a")
        templateList.append((row.sentiment, oldMutant, oldMutant, template, "old", text))
        templateList.append((row.sentiment, youngMutant, youngMutant, template, "young", text))
        
    elif '@adj' in template:   
        for placeholderValue in listOfPlaceholder:
            mutant = template.replace("@adj", placeholderValue).replace("@det", p.a(placeholderValue).split()[0])
            templateList.append((row.sentiment, mutant, mutant, template, placeholderValue, text))    
    
    counter += 1

outputData = pd.DataFrame(templateList)
outputData.to_csv("./asset/result-2.csv")

print("--- %s seconds ---" % (time.time() - start_time))
    



# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 20:05:50 2020

@author: MSI
"""


import spacy
import pandas as pd
import inflect


datasetLoc = "./asset/test.csv"
gawLoc =  './asset/gaw.txt'

df = pd.read_csv(datasetLoc, header=None, sep='\t')
df.columns = ['sentiment', 'review']

gaw = pd.read_csv(gawLoc, sep=",", header=None)
gaw.columns = ['x', 'y']
gaw = gaw['x'].append(gaw['y']).reset_index(drop=True)
gawList = gaw.to_list()
nlp = spacy.load('en_core_web_sm')
p = inflect.engine()
 
from spacy.matcher import Matcher
matcher = Matcher(nlp.vocab)
  
def isValidAdj(spacySpan):
    if spacySpan.root.head.text in gaw.to_list() or spacySpan[0].like_num:
        return True
    else:
        return False

def addPattern(listOfAdj, matcher):
    for element in listOfAdj:
        pattern = [{"LOWER": element}]
        matcher.add(element, None, pattern)
    pattern = [{"LIKE_NUM": True}, {"LOWER": "years"}, {"LOWER": "old"}]
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
    else:
        if spacySpan.root.i != 0:
            beginning = spacyDoc[0:spacySpan.start]
            ending = spacyDoc[spacySpan.end:]
            template = beginning.text + " {} ".format("@ageNumber") + ending.text
        else:
            ending = spacyDoc[spacySpan.end:]
            template = "{} ".format(placeholder) + ending.text
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
        if span.root.nbor(-1).text in ['a', 'an']:
            return True
    return False

def pipelineGenerateTempl(spacyDoc):
    template = ''
    adjSpan = searchAdj(spacyDoc)
    if adjSpan != None:
        template = createTempl(adjSpan)
    return template
        
listOfAdj = ['old', 'young', 'older', 'younger', 'oldest', 'youngest', 'elder', 'eldest']
addPattern(listOfAdj, matcher)  

#test case
doc1 = nlp("he goes to school everyday")
doc = nlp("the old man is becoming friend with the other younger guy")
text =  "Tagline: the lucky ones died...before watching this. I've never watched a Bulgarian movie from 1920's, so I can't say this is the worst movie ever made, but it surely is the worst movie I've ever watched. I can't almost remember it. All I can recall is a family of stupid people who don't do anything right. Their car has one wheel out of four stuck in the sand, so they decide that there's nothing to do and prepare to live the rest of their lives there. Then there's an old man who is aware of the existence of a band of cannibals in the whereabouts but has never considered the idea to report the fact to the police. And, speaking of the police...if those freaks have lived around there eating humans for years, lots of people must have disappeared...how come the sheriff didn't suspect anything? But I gave up asking questions after the first five minutes or so. The rest is bore. An hallucinated unbelievable bore. I will be merciful and won't speak about the dialogues. And the acting. And the effects. I will only mention the final scene, where the freak girl eliminates a snake (the snakes! they come out in the end, what the hell do they have to do with the story?) with a sniper-precise throw of a stone, demonstrating the full disregard of Mr. Craven for reality and for things that happen on planet Earth in general. I believe there have been riots when the film was first released in 1977. Even being eaten by a cannibal wouldn't be a fair punishment to the director for this attack on intelligence."

templateList = []
counter = 0
for index, row in df.iterrows():
    template = ''
    print("counter: {}".format(counter))
    text = row.review
    doc = nlp(text)
    template = pipelineGenerateTempl(doc)
    oldMutant = template.replace("@adjComparative", "older").replace("@adjSuperlative", "oldest").replace("@adj", "old").replace("@ageNumber", "62 years old").replace("@det", "an")
    youngMutant = template.replace("@adjComparative", "younger").replace("@adjSuperlative", "youngest").replace("@adj", "young").replace("@ageNumber", "23 years old").replace("@det", "a") 
    
    if template != '' and '@ageNumber' not in template:
        templateList.append((row.sentiment, oldMutant, oldMutant, template, "old", text))
        templateList.append((row.sentiment, youngMutant, youngMutant, template, "young", text))
    elif template != '' and '@ageNumber' in template:
        templateList.append((row.sentiment, oldMutant, oldMutant, template, "y years old", text))
        templateList.append((row.sentiment, youngMutant, youngMutant, template, "y years old", text))
    
    counter += 1

outputData = pd.DataFrame(templateList)
outputData.to_csv("./asset/result.csv")




    



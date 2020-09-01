import re
from string import punctuation
from string import digits
import pandas as pd
import numpy as np

import neuralcoref
# import en_core_web_sm
# nlp = en_core_web_sm.load()
import en_core_web_lg
nlp = en_core_web_lg.load()
coref = neuralcoref.NeuralCoref(nlp.vocab)
nlp.add_pipe(coref, name='neuralcoref')

symbol = ["!","#","$","%","*",",","/",":",";","@","^","_","`","|","~", ".", "?", ")", ">", "'m", "'s","'d","'ll","'ve","n't","'re", "~"]

NAME = "name"
PRONOUN = "pro"
SALUTATION = "sltn"
GAW = "gaw"

# NAME_PLACEHOLDER = "<" + NAME + ">"
# PRONOUN_PLACEHOLDER = "<" + PRONOUN + ">" 
# SALUTATION_PLACEHOLDER = "<" + SALUTATION + ">"
# GAW_PLACEHOLDER = "<" + GAW + ">"


# masculine pronoun
masculine_pronoun = ["he", "him", "his", "himself", "He", "Him", "His", "Himself", "HE"]

# feminine prononun
feminine_pronoun = ["she", "her", "her", "herself", "She","Her", "Her", "Herself", "SHE"]

_masculineToFemininePronoun = {}
_feminineToMasculinePronoun = {}
for _m, _f in zip(masculine_pronoun, feminine_pronoun) :
    _masculineToFemininePronoun[_m] = _f
    _feminineToMasculinePronoun[_f] = _m

# gender associated word
gaw = pd.read_csv("../../asset/gender_associated_word/masculine-feminine-person.txt")
masculine_gaw = gaw["masculine"].values
feminine_gaw = gaw["feminine"].values
    
# gender salutation word
masculine_salutation = ["Mr", "Mr.", "Mr.", "Mister", "Sir"]
feminine_salutation = ["Ms", "Ms.", "Mrs.", "Miss", "Madam"]

_masculineToFeminineSalutation = {}
_feminineToMasculineSalutation = {}
for _m, _f in zip(masculine_salutation, feminine_salutation) :
    _masculineToFeminineSalutation[_m] = _f
    _feminineToMasculineSalutation[_f] = _m
    

# load name from gender computer
gcm = pd.read_csv("../../asset/gender_computer/male_names_only.csv")
gcm = gcm.sample(frac=1, random_state=123)
mnames = gcm["name"].tolist()# # names from GC
gcf = pd.read_csv("../../asset/gender_computer/female_names_only.csv")
gcf = gcf.sample(frac=1, random_state=123)
fnames = gcf["name"].tolist()# # names from GC

# small name for debugging
# mnames = ["Alonzo", "Adam"] 
# fnames = ["Ebony", "Amanda"]
# mcountries = ["Trial", "Trial"]
# fcountries = ["Trial", "Trial"]

# countries = mcountries.copy()
# countries.extend(fcountries)
male_country = pd.read_csv("../../asset/gender_computer/unique_male_names_and_country.csv")
female_country = pd.read_csv("../../asset/gender_computer/unique_female_names_and_country.csv")

def getMaleNamesAndTheirCountries() :
    return male_country["name"].tolist(), male_country["country"].tolist()

def getFemaleNamesAndTheirCountries() :
    return female_country["name"].tolist(), female_country["country"].tolist()

def masculineToFemininePronoun(_m):
    return _masculineToFemininePronoun[_m]

def feminineToMasculinePronoun(_f):
    return _feminineToMasculinePronoun[_f]
    
def masculineToFeminineSalutation(male) :
    return _masculineToFeminineSalutation[male]

def feminineToMasculineSalutation(female) :
    return _feminineToMasculineSalutation[female]

def getMaleNamesFromGenderComputer(N=30) :
    return mnames[:N]

def getFemaleNamesFromGenderComputer(N=30) :
    return fnames[:N]

def getMasculineGenderAssociatedWord() :
    return masculine_gaw

def getFeminineGenderAssociatedWord() :
    return feminine_gaw



def removeHtmlTags(text):
    """Remove html tags from a string"""
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

def removeHex(text):
    """Remove hex from a string"""
    text = text.encode().decode('unicode_escape')
    return re.sub(r'[^\x00-\x7f]',r'', text)

def removeBackslash(text):
    """Remove backslash from a string"""
    return text.replace("\\", "")

def convertParticles(string):
    # function to delete fix particle spacing after text splitting
    # output:
        # string -> string
    
    return string.replace(" 'm", "'m").replace(" 's", "'s").replace(" 'd", "'d").replace(" 've", "'ve").replace(" n't", "n't").replace(" 're", "'re").replace(" - ", "")


def combineText(sentenceList):
    # function to combine string in a list of string
    # input:
        # sentenceList -> list
    # output:
        # fulltext -> string
    
    fulltext = ''
    for element in sentenceList:
        tempElement = nlp(element)
        for token in tempElement:
            if token.text in symbol:
                fulltext = fulltext + ''.join(token.text)
            else:
                fulltext = fulltext + ' ' + ''.join(token.text)        
    fulltext = convertParticles(fulltext.strip())
    return fulltext


def restructureText(text):
    # function to combine string in a list of string
    # input:
        # sentenceList -> list
    # output:
        # fulltext -> string
    
    fulltext = ''
    doc = nlp(text)
    for token in doc:
#             print(token)
        if token.text in symbol:
            fulltext = fulltext + token.text
        else:
            fulltext = fulltext + ' ' + token.text
    fulltext = convertParticles(fulltext.strip())
    return fulltext


def preprocessText(text):
    text = removeHtmlTags(text)
    text = removeHex(text)
    text = removeBackslash(text)
    text = restructureText(text)
    return text

def isInMasculinePronoun(text) :
    return text in masculine_pronoun
    
def isInFemininePronoun(text):
    return text in feminine_pronoun

def isInMasculineSalutation(text):
    return text in masculine_salutation

def isInFeminineSalutation(text):
    return text in feminine_salutation
    
def isInMasculineGenderAssosiatedWord(text):
    return text in getMasculineGenderAssociatedWord()

def isInFeminineGenderAssosiatedWord (text):
    return text in getMasculineGenderAssociatedWord()

def tag(text):
    return "<" + text + ">"

def getPronounPlaceholders(placeholders) :
    pronoun_placeholders = []
    for placeholder in placeholders :
        if "<" + PRONOUN in placeholder:
            pronoun_placeholders.append(placeholder)
            
    return pronoun_placeholders
        


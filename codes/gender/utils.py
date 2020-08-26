import re
from string import punctuation
from string import digits
import pandas as pd
import numpy as np


import neuralcoref
import en_core_web_sm
nlp = en_core_web_sm.load()
# import en_core_web_lg
# nlp = en_core_web_lg.load()
coref = neuralcoref.NeuralCoref(nlp.vocab)
nlp.add_pipe(coref, name='neuralcoref')

symbol = ["!","#","$","%","*",",","/",":",";","@","^","_","`","|","~", ".", "?", ")", ">", "'m", "'s","'d","'ll","'ve","n't","'re", "~"]

# masculine pronoun
masculine_pronoun = ["he", "him", "his", "himself", "He", "Him", "His", "Himself", "HE"]

# feminine prononun
feminine_pronoun = ["she", "her", "her", "herself", "She","Her", "Her", "Herself", "SHE"]

# gender associated word
gaw = pd.read_csv("../../data/gender_associated_word/masculine-feminine-person.txt")
masculine_gaw = gaw["masculine"].values
feminine_gaw = gaw["feminine"].values
    
# gender salutation word
masculine_salutation = ["Mr", "Mr.", "Mr.", "Mister", "Sir"]
feminine_salutation = ["Ms", "Ms.", "Mrs.", "Miss", "Madam"]


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
    return text in masculine_gaw

def isInFeminineGenderAssosiatedWord (text):
    return text in feminine_gaw

def tag(text):
    return "<" + text + ">"


from string import punctuation
from string import digits

import en_core_web_lg
import neuralcoref
nlp = en_core_web_lg.load()
coref = neuralcoref.NeuralCoref(nlp.vocab)
nlp.add_pipe(coref, name='neuralcoref')

symbol = ["!","#","$","%","*",",","/",":",";","@","^","_","`","|","~", ".", "?", ")", ">", "'m", "'s","'d","'ll","'ve","n't","'re", "~"]

def removeHtmlTags(text):
    """Remove html tags from a string"""
    import re
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

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
    text = removeBlackslash(text)
    text = restructureText(text)
    return text
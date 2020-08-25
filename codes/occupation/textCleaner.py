# -*- coding: utf-8 -*-
import spacy
import neuralcoref
from io import StringIO
from html.parser import HTMLParser
from pycorenlp import StanfordCoreNLP
from string import punctuation
from string import digits
symbol = ["!","#","$","%","*",",","/",":",";","@","^","_","`","|","~", ".", "?", ")", ">", "'m", "'s","'d","'ll","'ve","n't","'re", "~"]
nlp = spacy.load('en_core_web_lg')
neuralcoref.add_to_pipe(nlp)

def convertParticles(string):
    # function to delete fix particle spacing after text splitting
    # output:
        # string -> string
    
    return string.replace(" 'm", "'m").replace(" 's", "'s").replace(" 'd", "'d").replace(" 've", "'ve").replace(" n't", "n't").replace(" 're", "'re")

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


class MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs= True
        self.text = StringIO()
    def handle_data(self, d):
        self.text.write(d)
    def get_data(self):
        return self.text.getvalue()

def stripTags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()


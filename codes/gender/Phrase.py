import pandas as pd
from utils import nlp
from utils import isInMasculinePronoun, isInFemininePronoun
from utils import isInMasculineSalutation, isInFeminineSalutation
from utils import isInMasculineGenderAssosiatedWord, isInFeminineGenderAssosiatedWord 

# contain a word and its location inside the sentence
# The location is indicated by start char and end char
class Phrase: 
    phrase = ""
    tokens = None
    
    def __init__(self, phrase) :
        self.phrase = phrase 
        self.tokens = None
        self.gender = ""
        self.salutation = ""
        self.gender_associated_word = ""
        
        
    def __str__(self) :
        return self.phrase
    
    def __repr__(self) :
        return self.phrase
        
    def getPhrase(self):
        return self.phrase

    def getTokens(self) :
        if not self.tokens :
            self.tokens = nlp(self.phrase)
        return self.tokens

    def isGenderPronoun(self) :
        if isInMasculinePronoun(self.phrase):
            self.gender = "male"
            return True
        if isInFemininePronoun(self.phrase):
            self.gender = "female"
            return True
        return False
    
    def getGender(self) :
        return self.gender
    
    def isHasSalutation(self):
        if not self.tokens :
            self.tokens = nlp(self.phrase)
        tokens = self.tokens
        for token in tokens:
            if isInMasculineSalutation(token.text):
                self.salutation = token.text
                self.gender = "male"
                return True
        for token in tokens:
            if isInFeminineSalutation(token.text):
                self.salutation = token.text
                self.gender = "female"
                return True
        return False

    def getSalutation(self):
        return self.salutation
        
    def isContainGenderAssociatedWord(self):
        if not self.tokens :
            self.tokens = nlp(self.phrase)
        tokens = self.tokens
        for token in tokens:
#             print(token.text, token.pos_, token.dep_)
            if token.pos_ == "NOUN" and token.dep_ == "ROOT" :
                if isInMasculineGenderAssosiatedWord(token.text):
                    self.gender_associated_word = token.text
                    return True
                if isInFeminineGenderAssosiatedWord(token.text):
                    self.gender_associated_word = token.text
                    return True
        return False

    def getGenderAssociatedWord(self) :
        return self.gender_associated_word
        
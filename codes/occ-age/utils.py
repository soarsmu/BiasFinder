# -*- coding: utf-8 -*-
import spacy
import neuralcoref
from io import StringIO
from html.parser import HTMLParser
from pycorenlp import StanfordCoreNLP
from string import punctuation
from string import digits
import re
import pandas as pd
import numpy as np
import inflect

p = inflect.engine()

nlp_wrapper = StanfordCoreNLP('http://localhost:9000')

symbol = ["!","#","$","%","*",",","/",":",";","@","^","_","`","|","~", ".", "?", ")", ">", "'m", "'s","'d","'ll","'ve","n't","'re", "~"]

# nlp = spacy.load('en_core_web_sm')
nlp = spacy.load('en_core_web_lg')
neuralcoref.add_to_pipe(nlp)

### from textCleaner.py
#
#
#
#

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

### from textSplitter.py
#
#
#
#


class textSplitter(object):
    # class to split the sentence resulted from Stanford Core NLP
    
    def __init__(self, text):
        # input:
            # text -> string of multiple sentences
        self.originalText = text
        self.text = nlp_wrapper.annotate(text,
			properties={
				'annotators' : 'pos',
				'outputFormat' : 'json',
				'timeout' : 100000,
			})

    def getNumberOfSentences(self):
        # return the number of sentence in the text -> integer
        return len(self.text['sentences'])
    
    def getNumberOfTokens(self, index):
        # input:
            # index -> integer
        # output:
            # return the number of token in the sentence[index] -> integer
    	return len(self.text['sentences'][index]['tokens'])
    
    def getSentence(self, index):
        # rearrange the token of sentece[index] in json format to string format
        # input:
            # index -> integer
        # output:
            # sentence -> string
        
        sentence = []
        sentence = ''
        for i in range(self.getNumberOfTokens(index)):
            tempString = self.text['sentences'][index]['tokens'][i]['originalText']
            if tempString in symbol:                        
                sentence = sentence + ''.join(tempString)
            else:
                sentence = sentence + ' ' + ''.join(tempString)
        sentence = sentence[1:].replace(" ( ", " (").replace(" < ", " <")
        return sentence
    
    def getAllSentences(self):
        # return a list of sentence 
        
        sentenceList = []
        for i in range(self.getNumberOfSentences()):
            temp_string = self.getSentence(i)
            sentenceList.append(temp_string)
        return sentenceList


### from textAnnotator.py
#
#
#
#


class annotatedText:
    # Class to represent annotated text resulted from StandfordCoreNLP
    
    def __init__(self, text):
    # input:
        # text -> string
        
        self.originalText = text
        self.text = nlp_wrapper.annotate(text,
            properties={
                'ner.applyNumericClassifiers' : 'false',
                'ner.useSUTime' : 'false',
                'ner.applyFineGrained': 'false',
                'annotators': 'ner, pos',
                'outputFormat': 'json',
                'timeout': 9999999999,
            })
    def getOriginalText(self):
        # return string of original text -> string
        return self.originalText
    
    def getFullAnnotatedText(self):
        # return the annotated text in json format
        return self.text
    
    def getAnnotatedSentence(self, sentenceIndex):
        # input:
            # sentenceIndex -> integer
        # output:
            # the annotated sentence -> string
        return self.text['sentences'][sentenceIndex]
    
    def getNumberOfSentence(self):
        # output:
            # return the number of sentence in a text -> integer
        return len(self.getFullAnnotatedText()['sentences'])
    
    def getNumberOfToken(self, sentenceIndex):
        # input:
            # sentenceIndex -> integer
        # output:
            # return the number of token in sentence[sentenceIndex] -> integer
        return len(self.getAnnotatedSentence(sentenceIndex)['tokens'])
    
    def getNumberOfEntity(self, sentenceIndex):
        # input:
            # sentenceIndex -> integer
        # output:
            # return the number of entity in integer that exist in a sentence[sentenceIndex] -> integer
        return len(self.getAnnotatedSentence(sentenceIndex)['entitymentions'])
    
    def getName(self, sentenceIndex, checkedName):
        # input:
            # sentenceIndex -> integer
        # output:
            # return a list of unique name in a sentence[sentenceIndex] if name exist in the sentence, otherwise return empty list
        
        name = None
        offsetBegin = 0
        offsetEnd = 0
        for i in range(self.getNumberOfEntity(sentenceIndex)):
            if self.getAnnotatedSentence(sentenceIndex)['entitymentions'][i]['ner'] == 'PERSON':
                end = self.getAnnotatedSentence(sentenceIndex)['entitymentions'][i]['tokenEnd']
                if end < (self.getNumberOfToken(0)):
                    tokenAferName = self.getAnnotatedSentence(sentenceIndex)['tokens'][end]['originalText']
                else:
                    tokenAferName = self.getAnnotatedSentence(sentenceIndex)['tokens'][end-1]['originalText']
                tempName = self.getAnnotatedSentence(sentenceIndex)['entitymentions'][i]['text']
                if tokenAferName != "'s" and  tempName not in checkedName and punctuation not in tempName:
                    name = self.getAnnotatedSentence(sentenceIndex)['entitymentions'][i]['text'].strip(punctuation).strip(digits)
                    offsetBegin = self.getAnnotatedSentence(sentenceIndex)['entitymentions'][i]['characterOffsetBegin']
                    offsetEnd = self.getAnnotatedSentence(sentenceIndex)['entitymentions'][i]['characterOffsetEnd']
                    break
        return (name, offsetBegin, offsetEnd)

    def getTokenIndex(self, sentenceIndex, token):
        # input:
            # token -> string
            # token is assumed always exist in the text
            # sentenceIndex -> integer
        # output:
            # i -> integer
            # i is index of a token in the sentence
            # if token is not available in the sentence, return -999
        
        token_spacy_object = nlp(token)
        token = token_spacy_object[0:].root.text
        for i in range(self.getNumberOfToken(sentenceIndex)):
            if self.getAnnotatedSentence(sentenceIndex)['tokens'][i]['originalText'].strip(punctuation).strip(digits) == token:
                return i
        return -999
    
    def getPosTag(self, sentenceIndex, token):
        # input:
            # token -> string
            # token is assumed always exist in the text
            # sentenceIndex -> integer
        # output:
            # return postag of token -> string
            # if token index is invalid, will return empty string
        
        token_spacy_object = nlp(token)
        token = token_spacy_object[0:].root.text
        token_index = self.getTokenIndex(sentenceIndex, token)
        if token_index == -999:
            return ''
        return self.getAnnotatedSentence(sentenceIndex)['tokens'][token_index]['pos']
            
    def checkPosTag(self, sentenceIndex, name):
        # input:
            # sentenceIndex -> integer
            # name -> string
            # name must exist in the string, otherwise the function will be invalid
        # output:
            # True if the postag of token is 'NN' (singular noun) or  'NNS' (plural noun)
        nameSpacyObj = nlp(name)
        nameString = nameSpacyObj[0:].root.text
        if self.getPosTag(sentenceIndex, nameString) == 'NNP':
            return True
        else:
            return False

class annotatedTextOcc(annotatedText):
    
    def __init__(self, text) :
        self.originalText = text
        self.text = nlp_wrapper.annotate(text,
        properties={
            'ner.applyNumericClassifiers' : 'false',
            'ner.useSUTime' : 'false',
            'ner.applyFineGrained': 'true',
            'annotators': 'ner, pos',
            'outputFormat': 'json',
            'timeout': 9999999999,
        })
    
    def getName(self, sentence_index):
        # input:
            # sentence_index -> integer
        # output:
            # return a list of unique name in a sentence[sentence_index] if name exist in the sentence, otherwise return empty list
        
        name_list = []
        for i in range(self.getNumberOfEntity(sentence_index)):
            if self.getAnnotatedSentence(sentence_index)['entitymentions'][i]['ner'] == 'PERSON':
                name = self.getAnnotatedSentence(sentence_index)['entitymentions'][i]['text'].strip(punctuation).strip(digits)
                name_list.append(name)
        return name_list
    
    def getOccupation(self, sentence_index, checked_occ):
        # input:
            # sentence_index -> integer
        # output:
            # return a list of unique occupation in a sentence[sentence_index] if occupation exist in the sentence, otherwise return empty list
        occ = None
        offset_begin = 0
        offset_end = 0
        for i in range(self.getNumberOfEntity(sentence_index)):
            if self.getAnnotatedSentence(sentence_index)['entitymentions'][i]['ner'] == 'TITLE':
                # spacy_text = nlp(self.getAnnotatedSentence(sentence_index)['entitymentions'][i]['text'])
                # if spacy_text[-1].is_title or spacy_text[-1].is_upper:
                #     break
                # occ = self.getAnnotatedSentence(sentence_index)['entitymentions'][i]['text'].strip(punctuation).strip(digits)
                # if len(occ.split()) == 1 and occ not in occupation_list:
                #     occupation_list.append(occ)
                temp_occ = self.getAnnotatedSentence(sentence_index)['entitymentions'][i]['text'].strip(punctuation).strip(digits)
                if len(temp_occ.split()) == 1 and temp_occ not in checked_occ:
                    occ = temp_occ
                    offset_begin = self.getAnnotatedSentence(sentence_index)['entitymentions'][i]['characterOffsetBegin']
                    offset_end = self.getAnnotatedSentence(sentence_index)['entitymentions'][i]['characterOffsetEnd']
                    break
        return (occ, offset_begin, offset_end)
    
    def checkValidMutant(self, name_list, occ):
        # input: 
            # name_list -> list of string
            # occupation_list -> string
        # output:
            # return True if name_list AND occupation_list not empty
        if len(name_list) > 0 and occ != None:
            return True
        else:
            return False
    
    def getTokenIndex(self, sentence_index, token):
        # input:
            # token -> string
            # token is assumed always exist in the text
            # sentence_index -> integer
        # output:
            # i -> integer
            # i is index of a token in the sentence
            # if token is not available in the sentence, return -999
        
        for i in range(self.getNumberOfToken(sentence_index)):
            if self.getAnnotatedSentence(sentence_index)['tokens'][i]['originalText'].strip(punctuation).strip(digits) == token:
                return i
        return -999
    
    def getPosTag(self, sentence_index, token):
        # input:
            # token -> string
            # token is assumed always exist in the text
            # sentence_index -> integer
        # output:
            # return postag of token -> string
            # if token index is invalid, will return empty string
        
        token_index = self.getTokenIndex(sentence_index, token)
        if token_index == -999:
            return ''
        return self.getAnnotatedSentence(sentence_index)['tokens'][token_index]['pos']
            
    def checkPosTag(self, sentence_index, occupation):
        # input:
            # sentence_index -> integer
            # temp_occ -> string
            # temp_occ must exist in the string, otherwise the function will be invalid
        # output:
            # True if the postag of token is 'NN' (singular noun) or  'NNS' (plural noun)

        if self.getPosTag(sentence_index, occupation) == 'NN' or self.getPosTag(sentence_index, occupation) == 'NNS':
            return True
        else:
            return False


### from solveCoreferenceAge.py
#
#
#
#

def createCorefDict(spacyText):
    # input:
        # spacyText -> spacy doc
    # output:
        # corefSpanDict -> dictionary of coref cluster object
        # original coref cluster output from neuralcoref (._.coref_clusters): [the director: [the director, he, himself]]
        # createCorefDict output: {the director: he, himself}
    
    corefSpanDict = {}
    for i in range(len(spacyText._.coref_clusters)):
        mentionList = spacyText._.coref_clusters[i].mentions[1:]
        corefSpanDict[spacyText._.coref_clusters[i].main] = mentionList
    return corefSpanDict

def checkEntityCoref(corefSpanDict, entity):
    # check if an occupation has coreference or not
    # input:
        # corefSpanDict -> dictionary output from createCorefDict function
        # entity -> spacy span
    # output:
        # return true if it is exist, false otherwise
        
    entityList = entity.text.split()
    entityList.append(entity.text)
    occupationCoref = False
    for key in corefSpanDict.keys():
        if key.root.text in entityList:
            occupationCoref = True
    return occupationCoref

def createEntityCorefDict(corefSpanDict, entity, corefFound):
    # create dictionary of occupation coreference index in the text
    # input:
        # corefSpanDict -> dictionary output from createCorefDict function
        # entity -> spacy span
        # corefFound -> boolean
    # output:
        # indexDict -> dictionary of occupation coreference
        # e.g: {he: 9, himself: 14} where 9 and 14 indicate the token index in the text
        
    indexDict = {}
    if corefFound:
        entityList = entity.text.split()
        entityList.append(entity.text)    
        for element in corefSpanDict.keys():
            if element.text in entityList:
                temp_list = corefSpanDict[element]
                for token in temp_list:
                    indexDict[token[0].i] = token
    return indexDict

def pipelineCheckCoref(entity, fulltext):
    # input:
        # entity: spacy token
        # fulltext: spacy doc
    # output:
        # True if coref is found, otherwise False

    entitySpan = nlp(entity.text)[0:]
    corefSpanDict = createCorefDict(fulltext)
    isCorefFound = checkEntityCoref(corefSpanDict, entitySpan)
    indexDict = createEntityCorefDict(corefSpanDict, entitySpan, isCorefFound)
    if len(indexDict) > 0:
        return True, indexDict
    else:
        return False, indexDict


def pipelineSolveCoref(fulltext, indexDict, entity):
    # function to solve name coreference in the text
    # input:
        # fulltext -> spacy doc
        # indexDict -> dictionary
        # name -> spacy span
    # output:
        # solvedCorefText -> string
    
    tempString = ''
    solvedCorefText = ''
    
    entityList = entity.text.split()
    entityList.append(entity.text)

    for token in fulltext:
        if token.i < len(fulltext)-1: 
            if token.i in indexDict.keys():
                if token.text in entityList:
                    tempString  = '@ADJ ' + token.text
                else:
                    tempString = token.text
            else:
                tempString = token.text
        else:
            if token.i in indexDict.keys():
                if token.text in entityList:
                    tempString  = '@ADJ ' + token.text
                else:
                    tempString = token.text
            else:
                tempString = token.text           
        if tempString in symbol:
            solvedCorefText = solvedCorefText + ''.join(tempString)
        else:
            solvedCorefText = solvedCorefText + ' ' + ''.join(tempString)                
        solvedCorefText = convertParticles(solvedCorefText.strip())
    return  solvedCorefText
    

### from solveCoreferenceOcc.py
#
#
#
#



def checkEntityCoref(corefSpanDict, entity):
    # check if an occupation has coreference or not
    # input:
        # corefSpanDict -> dictionary output from createCorefDict function
        # entity -> spacy span
    # output:
        # return true if it is exist, false otherwise
        
    occupationFound = False
    for key in corefSpanDict.keys():
        if entity.root.text == key.root.text:
            occupationFound = True
    return occupationFound

def createEntityCorefDictOcc(corefSpanDict, entity, corefFound):
    # create dictionary of occupation coreference index in the text
    # input:
        # corefSpanDict -> dictionary output from createCorefDict function
        # entity -> spacy span
        # corefFound - > boolean
    # output:
        # indexDict -> dictionary of occupation coreference
        # e.g: {he: 9, himself: 14} where 9 and 14 indicate the token index in the text
    
    indexDict = {}
    if corefFound:   
        for element in corefSpanDict.keys():
            if element.text == entity.text:
                temp_list = corefSpanDict[element]
                for token in temp_list:
                    indexDict[token[0].i] = token
    return indexDict



def pipelineSolveCorefOcc(fulltext, indexDict, entity):
    # function to solve name coreference in the text
    # input:
        # fulltext -> spacy doc
        # indexDict -> dictionary
        # name -> spacy span
    # output:
        # solvedCorefText -> string
    
    tempString = ''
    solvedCorefText = ''     
    for token in fulltext:
        if token.i in indexDict.keys():
            if token.tag_ == 'PRP$' and token.pos_ == "DET" and token.text.lower() in ["his"]:
                tempString  = '@CorefPronounHIS'
            elif token.tag == 'PRP$' and token.pos_ == "DET" and token.text.lower() in ["her"]:
                tempString = '@CorefPronounHer'
            elif token.tag_ == 'PRP' and token.pos_ == "PRON" and token.text.lower() in ["she", "herself"]:
                tempString = '@CorefPronounShe'
            elif token.tag_ == 'PRP' and token.pos_ == "PRON" and token.text.lower() in ["he", "himself"]:
                tempString = '@CorefPronounHe'
            elif token.text.lower() == entity.text:
                tempString = '@CorefOCCUPATION'
            else:
                tempString = token.text
        
        else:
            tempString = token.text
                    
        if tempString in symbol:
            solvedCorefText = solvedCorefText + ''.join(tempString)
        else:
            solvedCorefText = solvedCorefText + ' ' + ''.join(tempString)     
        solvedCorefText = convertParticles(solvedCorefText.strip())
    return solvedCorefText


### from repairName.py
#
#
#
#

def getNameIndex(text, name):
    # function to get the index of occupation in a spacy doc
    # input:
        # text -> spacy doc
        # name -> spacy span
    # output: 
        # if token found in the spacy doc, return outputName -> spacy span
        # otherwise return 999
        
    outputName = None
    nameList = name.text.split()
    for token in text:
        if len(nameList) == 1:
            if token.text == nameList[0]:
                outputName = text[token.i:][0:1]
                break
        else:
            if token.text == nameList[0]:
                offset_begin = token.idx
            elif token.text == nameList[-1]:
                tempOffset = token.idx
                tempLen = len(token.text)
                offsetEnd = tempOffset + tempLen
                outputName = text.char_span(offset_begin, offsetEnd)
                break
    return outputName


def deletePROPN(text, name):
    # delete pronoun like Mr, Mrs, etc before name entity
    # input:
        # text -> spacy doc
        # doc -> spacy span
    # output:
        # repairedText -> string
        
    repairedText = ''
    for token in text:
        if token.i < (len(text)-1):
            if text[token.i+1].text == name[0].text:
                if token.pos_ != 'PROPN':
                    if token.text in symbol:
                        repairedText = repairedText + ''.join(token.text)
                    else:
                        repairedText = repairedText + ' ' + ''.join(token.text)
            else:
                if token.text in symbol:
                    repairedText = repairedText + ''.join(token.text)
                else:
                    repairedText = repairedText + ' ' + ''.join(token.text)
        else:
            if token.text in symbol:
                repairedText = repairedText + ''.join(token.text)
            else:
                repairedText = repairedText + ' ' + ''.join(token.text)
    repairedText = convertParticles(repairedText.strip())
    return repairedText

### from repairOcc.py
#
#
#
#

def getOccIndex(text, occupation):
    # function to get the index of occupation in a spacy doc
    # input:
        # text -> spacy doc
        # occupation -> spacy token
    # output: 
        # if token found in the spacy doc, return token -> spacy token
        # otherwise return None
        
    for token in text:
        if token.text == occupation.text:
            return token
    return None

def modifyOcc(occupation):
    # function to delete noun phrase between determiner and occupation
    # e.g: a high school student becomes a student
    # input:
        # occupation -> spacy token
    #output:
        # repairedOcc -> string
        
    repairedOcc = ''
    for token in occupation.children:
        if token.pos_ not in ['NOUN'] and token.i < occupation.i:
            for subtoken in token.children:
                if subtoken.pos_ not in ['NOUN'] and subtoken.i < token.i:
                    repairedOcc = repairedOcc + ' ' + ''.join(subtoken.text)
            repairedOcc = repairedOcc + ' ' + ''.join(token.text)
            for subtoken in token.children:
                if subtoken.pos_ not in ['NOUN'] and subtoken.i > token.i:
                    repairedOcc = repairedOcc + ' ' + ''.join(subtoken.text)

    repairedOcc = repairedOcc.strip()
    repairedOcc = '{} {}'.format(repairedOcc, occupation.text)
    
    repairedOccSpacyObj = nlp(repairedOcc)
    newOccupation = getOccIndex(repairedOccSpacyObj, occupation)
    if newOccupation != None:
        repairedOcc = ''
        for token in repairedOccSpacyObj:
            if token.i == (newOccupation.i - 1) and token.text in ['a', 'an']:
                repairedOcc = repairedOcc + ' ' + ''.join("@DetAAN")
            else:
                repairedOcc = repairedOcc + ' ' + ''.join(token.text)
        
    return repairedOcc.strip()

def repairSentence(text, occupation):
    # funtion to repair sentence by deletin phrase between determiner and occupation
    # input:
        # text -> spacy doc
        # occupation -> spacy token
    # output:
        # repairedText -> repaired spacy doc
        # newOccupation -> spacy token

    repairedText = None
    newOccupation = ''
    
    tempDict = {}
    tempDict[occupation] = occupation.i
    for token in occupation.subtree:
        tempDict[token] = token.i
        
    beginSpanStopIndex = min(tempDict.values())
    endSpanStartIndex = occupation.i + 1
    
    beginSpan = text[0:beginSpanStopIndex].text
    endSpan = text[endSpanStartIndex:].text
    
    occString = modifyOcc(occupation)
    
    repairedText = nlp(('{} {} {}'.format(beginSpan,occString,endSpan)).strip())
    newOccupation = getOccIndex(repairedText, occupation)
    return (repairedText, newOccupation)



### from mutantTextAge.py
#
#
#
#

def searchMutantSentence(sentenceList, checkedName, checkpoint, isLast, updateCheckpoint):
    # search the mutantable sentence in a list of sentence
    # input:
        # sentenceList -> list of sentence in string
        # checkedName -> tuple of checked name in string
        # checkpoint -> integer
        # isLast ->  boolean
        # updateCheckpoint -> boolean
    # output:
        # mutantSentenceIndex -> integer
        # name -> spacy token
        # isNameFound -> boolean
        # checkedName -> tuple of checked name in string
        # checkpoint -> integer
        # isLast -> boolean
        
    # process each sentence in the list, if occupation found, break the for loop (ignore the remaining sentences
    # because we already get a placeholder
    mutantSentenceIndex = 999
    isNameFound = False
    sentenceCounter = checkpoint
    name = None
    for i in range(checkpoint, len(sentenceList)):
        text = annotatedText(sentenceList[i])
        for sentenceIndex in range(text.getNumberOfSentence()):
            name, offsetBegin, offsetEnd = text.getName(sentenceIndex, checkedName)
            if name != None and name not in checkedName:
                nameList = name.split()
                if len(nameList) < 3:
                    if text.checkPosTag(sentenceIndex, name):
                        mutantSentence = nlp(text.getOriginalText())
                        name = mutantSentence.char_span(offsetBegin, offsetEnd)
                        if name != None:
                            isNameFound = True
                            ######## add to get one template only
                            isLast = True
                            ########

                            mutantSentenceIndex = sentenceCounter
                            if updateCheckpoint == True:
                                checkedName = checkedName + (name.text,)
                                checkpoint = mutantSentenceIndex
                            break              
        if isNameFound == True and name != None:
            break
        sentenceCounter += 1
    if sentenceCounter == len(sentenceList):
        checkpoint = sentenceCounter - 1
        isLast = True
    return mutantSentenceIndex, name, isNameFound, checkedName, checkpoint, isLast

def generateMutant(mutantSentence, name):
    # funtion to generate mutant sentece
    # input:
        # mutantSentence -> spacy doc
        # name -> integer
    # output:
        # mutantSentece -> spacy doc
    
    nameList = name.text.split()
    nameList.append(name.text)
    mutant = ''
    for token in mutantSentence:
        if token.i < len(mutantSentence)-1:
            if token.text in nameList and mutantSentence[token.i-1].text not in nameList:
                mutant = mutant + ' ' + ''.join("{} {}".format("@ADJ", token.text))
            else:
                mutant = mutant + ' ' + ''.join(token.text)
        
        else:
            if token.text in nameList and mutantSentence[token.i-1].text not in nameList:
                mutant = mutant + ' ' + ''.join("{} {}".format("@ADJ", token.text))
            else:
                mutant = mutant + ' ' + ''.join(token.text)
    
    return (nlp(mutant.strip()))

### from mutantTextOcc.py
#
#
#
#

from textAnnotator import annotatedTextOcc as at 
import spacy

def searchMutantSentenceOcc(sentenceList, checkedOcc, checkpoint, isLast, updateCheckpoint):
    # search the mutantable sentence in a list of sentence
    # input:
        # sentenceList -> list of sentence in string
        # checkedName -> tuple of checked name in string
        # checkpoint -> integer
        # isLast ->  boolean
        # updateCheckpoint -> boolean
    # output:
        # mutantSentence -> spacy doc
        # mutantSentenceIndex -> integer
        # occupation -> spacy token
        # isOccupationFound -> boolean
        # checkedOcc -> tuple of checked name in string
        # checkpoint -> integer
        # isLast -> boolean
        
    # process each sentence in the list, if occupation found, break the for loop (ignore the remaining sentences
    # because we already get a placeholder
    mutantSentenceIndex = 999
    isOccupationFound = False
    sentenceCounter = checkpoint
    occupation = None
    mutantSentence = None
    for i in range(checkpoint, len(sentenceList)):
        text = annotatedTextOcc(sentenceList[i])
        for sentenceIndex in range(text.getNumberOfSentence()):
            occ, offsetBegin, offsetEnd = text.getOccupation(sentenceIndex, checkedOcc)
            if occ != None and occ not in checkedOcc:
                nameList = text.getName(sentenceIndex)
                if text.checkValidMutant(nameList, occ) and isOccupationFound == False and text.checkPosTag(sentenceIndex, occ):
                    mutantSentence = nlp(text.getOriginalText())    
                    occupation = mutantSentence.char_span(offsetBegin, offsetEnd)
                    if occupation != None:
                        occupation = occupation[0]
                        isOccupationFound = True
                        ######## add to get one template only
                        isLast = True
                        ########
                        mutantSentenceIndex = sentenceCounter
                        if updateCheckpoint:
                            checkedOcc = checkedOcc + (occupation.text,)
                            checkpoint = mutantSentenceIndex
                        break
        
        if isOccupationFound == True and occupation != None:
            break
        sentenceCounter += 1
    if sentenceCounter == len(sentenceList):
        checkpoint = sentenceCounter - 1
        isLast = True
        
#     print(mutantSentence, mutantSentenceIndex, occupation, isOccupationFound, checkedOcc, checkpoint, isLast)
    return mutantSentence, mutantSentenceIndex, occupation, isOccupationFound, checkedOcc, checkpoint, isLast

def generateMutantOcc(mutantSentence, occupation):
    # funtion to generate mutant sentece
    # input:
        # mutantSentence -> spacy doc
        # occupation -> spacy token
    # output:
        # mutant -> spacy doc
        
    mutant = ''
    for token in mutantSentence:
        if token.text == occupation.text:
            mutant = mutant + ' ' + ''.join("@OCCUPATION")
        else:
            mutant = mutant + ' ' + ''.join(token.text)
    
    # mutant_text = spacy_fulltext.text[0:token_count] + ' @OCCUPATION ' + spacy_fulltext.text[token_count + (len(occupation.text))+2:]
    return (nlp(mutant.strip()))


### from mainPipelineAge.py
#
#
#
#



def generateMutantAge(inputText, checkedname, checkpoint, sentenceList, isLast):
    # main pipeline to process a review 
    # input:
        # inputText -> string (can be a single paragraph or multiple paragraph)
        # checkedname -> tuple of checked name in string
        # checkpoint -> integer
        # sentenceList -> list of string
        # isLast -> boolean
    # output:
        # mutantTemplate -> string
        # name -> spacy token
        # checkedname -> tuple of checked name in string
        # checkpoint -> integer
        # isLast -> boelan

#     print("X")

    mutantSentenceIndex, name, isNameFound, checkedname, checkpoint, isLast = searchMutantSentence(sentenceList, checkedname, checkpoint, isLast, False)
    mutantTemplate = None
    if isNameFound and name != None:
    
        repairedText = deletePROPN(nlp(inputText), name)
        sentenceList = textSplitter(repairedText).getAllSentences()
        
        mutantSentenceIndex, name, isNameFound, checkedname, checkpoint, isLast = searchMutantSentence(sentenceList, checkedname, checkpoint, isLast, True)
#         print('name found: ', name)
        
        if name != None and mutantSentenceIndex != 999:
            
            textAfterRepair = combineText(sentenceList)
            isCorefFound, indexCoref = pipelineCheckCoref(name, nlp(repairedText))
#             print(indexCoref)
            if isCorefFound:
                for j in indexCoref.values():
                    checkedname = checkedname + (j.text,)
#                 print("coref found")
                solvedCorefText = pipelineSolveCoref(nlp(textAfterRepair), indexCoref, name)
                sentenceList = textSplitter(solvedCorefText).getAllSentences()
                mutantSentence = sentenceList[mutantSentenceIndex]
                
                mutantSentenceTemplate = generateMutant(nlp(mutantSentence), name)
    
                copyOfSentenceList = sentenceList[:]
                copyOfSentenceList[mutantSentenceIndex] = mutantSentenceTemplate.text
                mutantTemplate = combineText(copyOfSentenceList)
                mutantTemplate = (convertParticles(mutantTemplate.replace("@ADJ @ADJ", "@ADJ").replace("( ", "(")))
            
            else:
#                 print("coref not found")
                sentenceList = textSplitter(textAfterRepair).getAllSentences()
                mutantSentence = sentenceList[mutantSentenceIndex]
                mutantSentenceTemplate = generateMutant(nlp(mutantSentence), name)
    
                copyOfSentenceList = sentenceList[:]
                copyOfSentenceList[mutantSentenceIndex] = mutantSentenceTemplate.text
                mutantTemplate = combineText(copyOfSentenceList)
                mutantTemplate = (convertParticles(mutantTemplate.replace("@ADJ @ADJ", "@ADJ").replace("( ", "(")))
        
#     else:
#         print("name not found")
    if mutantTemplate != None:
        if '@ADJ' not in mutantTemplate and isNameFound:
#             print("invalid mutant")
            mutantTemplate = ''
    mutantYoung = None
    mutantOld = None
    if mutantTemplate != None and name != None:
#         print("resulted template")
#         print(mutantTemplate)
        mutantYoung = mutantTemplate.replace("@ADJ", "young")
        mutantOld = mutantTemplate.replace("@ADJ", "old").replace('@DetAAN', (p.a('old').split()[0]))
    
#     print("XX")

        
    return (mutantYoung, mutantOld, mutantTemplate, name, checkedname, checkpoint, isLast)



def generateMultipleMutantAge(text):
    # input:
        # text -> string
    # output:
        # outputTuple -> tuple
    
#     strippedText = text.replace("<br />", '').strip()
    
    strippedText = preprocessText(text)
    
    # convert all the splitted sentence into list
    sentenceList = textSplitter(strippedText.strip()).getAllSentences()

    #rearrange the sentences to overcome sentence splitting issue
    tempString = ''
    for element in sentenceList:
        tempSpacyObj = nlp(element)
        for token in tempSpacyObj:
            if token.text in symbol:
                tempString = tempString + ''.join(token.text)
            else:
                tempString = tempString + ' ' + ''.join(token.text)
    
    sentenceList = textSplitter(tempString.strip()).getAllSentences()
    cleanedText = combineText(sentenceList)
    
    checkedname = ()
    checkpoint = 0
    outputTuple = ()
    isLast = False
    while not(isLast):
#         print("---------------------------")
#         print("sentence: {}".format(checkpoint))
        mutantTemplate = ''
        mutantYoung, mutantOld, mutantTemplate, name, checkedname, checkpoint, isLast = generateMutantAge(cleanedText, checkedname, checkpoint, sentenceList, isLast)
        if mutantTemplate != None and mutantYoung != None and mutantOld != None:
            outputTuple = outputTuple + ((mutantYoung, mutantTemplate, cleanedText, name, 'young'),)
            outputTuple = outputTuple + ((mutantOld, mutantTemplate, cleanedText, name, 'old'),)
    return outputTuple


### from mainPipelineOcc.py
#
#
#
#

def generateMutantsOcc(inputText, checkedOccupation, checkpoint, sentenceList, isLast):
    # main pipeline to process a review 
    # input:
        # inputText -> string (can be a single paragraph or multiple paragraph)
        # checkedOccupation -> tuple of checked name in string
        # checkpoint -> integer
        # sentenceList -> list of string
        # isLast -> boolean
    # output:
        # mutantTemplate -> string
        # occupation -> spacy token
        # checkedOccupation -> tuple of checked name in string
        # checkpoint -> integer
        # isLast -> boelan
    
    mutantSentence, mutantSentenceIndex, occupation, isOccupationFound, checkedOccupation, checkpoint, isLast = searchMutantSentenceOcc(sentenceList, checkedOccupation, checkpoint, isLast, True)

    mutantTemplate = None
    mutantSentenceTemplate = None
    if isOccupationFound and occupation != None and mutantSentence != None:
        repairedOccSentence, occupation = repairSentence(mutantSentence, occupation)
        if occupation != None and repairedOccSentence != None:
#             print("occupation found: {}".format(occupation))
            isCorefFound, indexCoref = pipelineCheckCoref(occupation, nlp(inputText))
            if isCorefFound:
                for j in indexCoref.values():
                    checkedOccupation = checkedOccupation + (j.text,)
#                 print("coref found")
                
                solvedCorefText = pipelineSolveCorefOcc(nlp(inputText), indexCoref, occupation)
                sentenceList = textSplitter(solvedCorefText).getAllSentences()
                
                mutantSentenceTemplate = generateMutantOcc(repairedOccSentence, occupation)
    
                copyOfSentenceList = sentenceList[:]
                copyOfSentenceList[mutantSentenceIndex] = mutantSentenceTemplate.text
                mutantTemplate = combineText(copyOfSentenceList)
                mutantTemplate = (convertParticles(mutantTemplate.replace("( ", "(")))
            
            else:
#                 print("coref not found")
                sentenceList = textSplitter(inputText).getAllSentences()
                mutantSentenceTemplate = generateMutantOcc(repairedOccSentence, occupation)
    
                copyOfSentenceList = sentenceList[:]
                copyOfSentenceList[mutantSentenceIndex] = mutantSentenceTemplate.text
                mutantTemplate = combineText(copyOfSentenceList)
                mutantTemplate = (convertParticles(mutantTemplate.replace("( ", "(")))
        
        # solve plural occupation word in the text
        occPlural = p.plural(occupation.text)
        mutantTemplate = re.sub(r'\b' + occPlural + r'\b', '@OccPLURAL', mutantTemplate)
#     else:
#         print("occupation not found")
    if mutantTemplate != None:
        if '@OCCUPATION' not in mutantTemplate and isOccupationFound:
#             print("invalid mutant")
            mutantTemplate = ''
#         print("resulted template")
#         print(mutantTemplate)

    return (mutantTemplate, occupation, checkedOccupation, checkpoint, isLast)

def generateMultipleMutantOcc(text, occupationPlaceholder):
    # input:
        # text -> string
        # occupationPlaceholder -> list of placeholder
    # output:
        # outputTuple -> tuple
    
#     strippedText = text.replace("<br />", '').strip()

    strippedText = preprocessText(text)

    # convert all the splitted sentence into list
    sentenceList = textSplitter(strippedText.strip()).getAllSentences()
    
    #rearrange the sentences to overcome sentence splitting issue
    tempString = ''
    for element in sentenceList:
        tempSpacyObj = nlp(element)
        for token in tempSpacyObj:
            if token.text in symbol:
                tempString = tempString + ''.join(token.text)
            else:
                tempString = tempString + ' ' + ''.join(token.text)
    
    sentenceList = textSplitter(tempString.strip()).getAllSentences()
    cleanedText = combineText(sentenceList)
    
    checkedOccupation = ()
    checkpoint = 0
    outputTuple = ()
    isLast = False
    tempList = []
    mutantOcc = ''
    while not(isLast):
#         print("---------------------------")
#         print("sentence: {}".format(checkpoint))
        mutantTemplate, occupation, checkedOccupation, checkpoint, isLast = generateMutantsOcc(cleanedText, checkedOccupation, checkpoint, sentenceList, isLast)
        if mutantTemplate != None:
            tempList = []
            for element in occupationPlaceholder:
                mutantOcc = ''.join(mutantTemplate.replace('@DetAAN', (p.a(element).split()[0])).replace('@OCCUPATION', element).replace('@CorefPronounHER', 'her').replace('@CorefPronounHIS', 'his').replace('@CorefPronounSHE', 'she').replace('@CorefPronounHE', 'he').replace("@CorefOCCUPATION", element))
                tempList.append((mutantOcc, mutantTemplate, cleanedText, occupation, element))
 
            if len(tempList) > 0:
                for element in tempList:
                    outputTuple = outputTuple + ((element),)
    return outputTuple


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
#     text = restructureText(text)
    return text

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
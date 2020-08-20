# -*- coding: utf-8 -*-
from textCleaner import convertParticles
from textCleaner import symbol
import spacy
import neuralcoref

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
    
    nlp = spacy.load('en_core_web_lg')
    neuralcoref.add_to_pipe(nlp)
    
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
    


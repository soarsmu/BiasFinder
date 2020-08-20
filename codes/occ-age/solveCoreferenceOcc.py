from textCleaner import convertParticles
from textCleaner import symbol
from solveCoreferenceAge import createCorefDict
from solveCoreferenceAge import pipelineCheckCoref
import spacy
import neuralcoref


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







    
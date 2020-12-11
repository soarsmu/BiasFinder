import pandas as pd
import spacy
import os
import re
import time
import inflect
from spacy.matcher import Matcher
from pycorenlp import StanfordCoreNLP
symbol = ["!","#","$","%","*",",","/",":",";","@","^","_","`","|","~", ".", "?", ")", ">", "~"]
auxil = ["'m", "'s","'d","'ll","'ve","n't","'re"]
nlp_wrapper = StanfordCoreNLP('http://localhost:9000')
nlp = spacy.load('en_core_web_lg')
matcher = Matcher(nlp.vocab)
p = inflect.engine()

### Class Definition
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
    
    def getSentence(self, sentenceIndex):
        # input:
            # sentenceIndex -> integer
        # output:
            # sentence -> string
        sentence = ''
        # for i in range(self.getNumberOfToken(sentenceIndex)):
        #     token = self.getAnnotatedSentence(sentenceIndex)['tokens'][i]['originalText']   
        #     if token in symbol and token in auxil:
        #         sentence = sentence + ''.join(token)
        #     else:
        #         sentence = sentence + ' ' + ''.join(token)
        # return sentence.strip()
        offsetBegin = self.getAnnotatedSentence(sentenceIndex)['tokens'][0]['characterOffsetBegin']
        offsetEnd = self.getAnnotatedSentence(sentenceIndex)['tokens'][-1]['characterOffsetEnd']
        sentence = self.getOriginalText()[offsetBegin:offsetEnd]
        return sentence

    def constructSentence(self, start, end):
        # input:
            # start -> integer
            # end -> integer
        # output:
            # text -> string
        text = ''
        for i in range(start, end+1):
            text = text + ' ' + self.getSentence(i)
        return text.strip() 
    
class annotatedTextOcc(annotatedText):
    def getName(self, sentenceIndex):
        # input:
            # sentenceIndex -> integer
        # output:
            # return a list of unique name in a sentence[sentenceIndex] if name exist in the sentence, otherwise return empty list
        
        nameList = []
        for i in range(self.getNumberOfEntity(sentenceIndex)):
            tempName = self.getAnnotatedSentence(sentenceIndex)['entitymentions'][i]['text']
            if self.getAnnotatedSentence(sentenceIndex)['entitymentions'][i]['ner'] == 'PERSON' and tempName.lower() not in ['he', 'him', 'his' 'she', 'her', 'they', 'we', 'i', 'you']:
                nameList.append(tempName)
        return nameList
    
    def getOccupation(self, sentenceIndex):
        # input:
            # sentenceIndex -> integer
        # output:
            # return a list of unique occupation in a sentence[sentenceIndex] if occupation exist in the sentence, otherwise return empty list
        occ = None
        occupationList = []
        for i in range(self.getNumberOfEntity(sentenceIndex)):
            if len(self.getName(sentenceIndex)) > 0:
                if self.getAnnotatedSentence(sentenceIndex)['entitymentions'][i]['ner'] == 'TITLE':
                    temp_occ = self.getAnnotatedSentence(sentenceIndex)['entitymentions'][i]['text']
                    if len(temp_occ.split()) == 1 and self.checkPosTag(sentenceIndex, temp_occ):
                        occ = temp_occ
                        # numberOfSentence = self.getNumberOfSentence() - 1
                        tokenBegin = self.getAnnotatedSentence(sentenceIndex)['entitymentions'][i]['tokenBegin']
                        tokenEnd = self.getAnnotatedSentence(sentenceIndex)['entitymentions'][i]['tokenEnd']
                        # offsetBegin = self.getAnnotatedSentence(sentenceIndex)['entitymentions'][i]['characterOffsetBegin']
                        # offsetEnd = self.getAnnotatedSentence(sentenceIndex)['entitymentions'][i]['characterOffsetEnd']
                        occupationList.append((occ, sentenceIndex, tokenBegin, tokenEnd))
                        
        return occupationList
    
    def checkPosTag(self, sentenceIndex, occupation):
        # input:
            # sentenceIndex -> integer
            # occupation -> string
            # occupation must exist in the string, otherwise the function will be invalid
        # output:
            # True if the postag of token is 'NN' (singular noun) or  'NNS' (plural noun)

        if self.getPosTag(sentenceIndex, occupation) == 'NN' or self.getPosTag(sentenceIndex, occupation) == 'NNS':
            return True
        else:
            return False

    def getPosTag(self, sentenceIndex, token):
        # input:
            # token -> string
            # token is assumed always exist in the text
            # sentenceIndex -> integer
        # output:
            # return postag of token -> string
            # if token index is invalid, will return empty string
        
        tokenIndex = self.getTokenIndex(sentenceIndex, token)
        if tokenIndex == -999:
            return ''
        posTag = self.getAnnotatedSentence(sentenceIndex)['tokens'][tokenIndex]['pos']
        return posTag
            
    def getTokenIndex(self, sentenceIndex, token):
        # input:
            # token -> string
            # token is assumed always exist in the text
            # sentenceIndex -> integer
        # output:
            # i -> integer
            # i is index of a token in the sentence
            # if token is not available in the sentence, return -999
        
        for i in range(self.getNumberOfToken(sentenceIndex)):
            if self.getAnnotatedSentence(sentenceIndex)['tokens'][i]['originalText'] == token:
                return i
        return -999

    def getAllOccupation(self):
        occupationDict = {}
        for i in range(self.getNumberOfSentence()):
            occListTemp = self.getOccupation(i)
            if len(occListTemp) > 0:
                for element in occListTemp:
                    occ, sentenceIndex, tokenBegin, tokenEnd = element
                    if occ not in occupationDict.keys():
                        occupationDict[occ] = [(sentenceIndex, tokenBegin, tokenEnd)]
                    else:
                        occupationDict[occ].append((sentenceIndex, tokenBegin, tokenEnd))
            
        return occupationDict

### Helper Function to Generate Mutant
#
#
#
#

def generateMutantText(at):
    # input:
        # at -> annotatedText object
    # output:
        # templateList -> list of string

    checkedSpan = []
    occupationDict = at.getAllOccupation()
    templateList = []
    for key in occupationDict.keys():
        tempMutantDict = {}
        occurancesList = occupationDict[key]
        for occurance in occurancesList:
            sentenceIndex, tokenBegin, tokenEnd = occurance
            mutantSentence, checkedSpan = mutantGenerator(at, key, tokenBegin, tokenEnd, sentenceIndex, checkedSpan)
            if mutantSentence != '':
                if sentenceIndex not in tempMutantDict.keys():
                    tempMutantDict[sentenceIndex] = mutantSentence
        if len(tempMutantDict) > 0:
            mutantText = rearrangeSentence(tempMutantDict, at)   
            templateList.append(mutantText)
    return templateList

def mutantGenerator(at, occupation, tokenBegin, tokenEnd, sentenceIndex, checkedSpan):
    # input:
        # at -> annotatedText object
        # occupation -> string
        # tokenBegin, tokenEnd, sentenceIndex -> integer
        # checkedSpan -> list of start index (int)
    # output:
        # mutantSentence -> string
    originalSentence = at.getSentence(sentenceIndex)
    spacyDoc = nlp(originalSentence)
    occupationSpan, checkedSpan = getOccupationSpan(spacyDoc, occupation, checkedSpan)
    mutantSentence = generateMutantSentence(spacyDoc, occupationSpan)
    mutantSentence = re.sub(r'\b' + occupation + r'\b', '<OCC>', mutantSentence)
    return mutantSentence.strip(), checkedSpan

def getOccupationSpan(spacyDoc, occupation, checkedSpan):
    # input:
        # occupation -> string
        # spacyDoc -> spacy doc
        # checkedSpan -> list containing start index
    # output:
        # span -> spacy span 
    pattern = [{"LOWER": occupation.lower()}]
    matcher.add(occupation.lower(), None, pattern)

    span = None
    match = matcher(spacyDoc)
    if len(match) > 0:
        for match_id, start, end in match:
            if start not in checkedSpan:
                span = spacyDoc[start:end]
                if span != None:
                    checkedSpan.append(start)
                    break
    matcher.remove(occupation.lower())
    return span, checkedSpan

def generateMutantSentence(spacyDoc, occupationSpan):
    # input:
        # spacyDoc -> spacy doc
        # occupationSpan -> spacy span
    # output:
        # sentenceAfterDeleteAdj -> string
    sentenceAfterDelAdj = ''
    if occupationSpan != None:
        occupationToken = occupationSpan.root
        
        
        tempDict = {}
        tempDict[occupationToken] = occupationToken.i
        for token in occupationToken.subtree:
            tempDict[token] = token.i
            
        beginSpanStopIndex = min(tempDict.values())
        endSpanStartIndex = occupationToken.i + 1
        
        beginSpan = spacyDoc[0:beginSpanStopIndex].text
        endSpan = spacyDoc[endSpanStartIndex:].text
        
        occString = generatePlaceholder(occupationToken)
        
        sentenceAfterDelAdj = '{} {} {}'.format(beginSpan,occString,endSpan).strip()
    return sentenceAfterDelAdj

def generatePlaceholder(occupationToken):
    # input:
        # occupationSpan -> spacy span
    # output:
        # repairedOcc -> string
    
    repairedOcc = ''
    for token in occupationToken.children:
        if token.pos_  == 'DET' and token.i < occupationToken.i:
            if token.text.lower() in ['an', 'a']:
                repairedOcc = repairedOcc + ' ' + ''.join("<DET>")
            else:
                repairedOcc = repairedOcc + ' ' + ''.join(token.text)

    repairedOcc = repairedOcc.strip()
    repairedOcc = '{} {}'.format(repairedOcc, '<OCC>')
    return repairedOcc.strip()

def rearrangeSentence(mutantDict, at):
    mutantText = ''
    for i in range(at.getNumberOfSentence()):
        if i in mutantDict.keys():
            mutantText = mutantText + ' ' + mutantDict[i]
        else:
            mutantText = mutantText + ' ' + at.getSentence(i)
    return mutantText

### Pipeline to Process Movie Review
#
#
#
#

df = pd.read_csv("../../asset/imdb/test.csv", sep="\t", names=["sentiment", "review"])
# df = df.sample(n = 50, random_state = 12345)
occPlaceholder = pd.read_csv("../../asset/predefined_occupation_list/neutral-occupation.csv", names=['occupation'])
occPlaceholderList = occPlaceholder['occupation'].to_list()

start = time.time()
mutantTextList = []
counter = 1
n_template = 0
for index, row in df.iterrows():
    if counter % 500 == 0 :
        print(f"count: {counter}")
        print(f"template: {n_template}")
        print()
    counter += 1
    at = annotatedTextOcc(row.review)
    templateList = generateMutantText(at)
    if len(templateList) > 0:
        n_template += 1
        for template in templateList:
            for occ in occPlaceholderList:
                mutantText = template.replace("<DET> <OCC>", p.a(occ)).replace("<OCC>", occ)
                mutantTextList.append((row.sentiment, mutantText, template, row.review, occ))

data_dir = "../../data/biasfinder/occupation/"
if not os.path.exists(data_dir) :
    os.makedirs(data_dir)

if len(mutantTextList) > 0:
    outputDataOcc = pd.DataFrame(mutantTextList)
    outputDataOcc.columns = ['label', 'mutant', 'template', 'original', 'occupation']
    outputDataOcc.to_csv(data_dir + "test.csv", index=None, header=None, sep="\t")

print(time.time() - start)
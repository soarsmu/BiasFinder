from string import punctuation
from string import digits
import spacy
from pycorenlp import StanfordCoreNLP
# nlp_wrapper = StanfordCoreNLP('10.4.0.15:9000')
nlp_wrapper = StanfordCoreNLP('http://localhost:9000')
nlp = spacy.load('en_core_web_sm')

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




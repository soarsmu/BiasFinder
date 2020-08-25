from pycorenlp import StanfordCoreNLP
from textCleaner import symbol
# nlp_wrapper = StanfordCoreNLP('172.17.0.1:9000')
# nlp_wrapper = StanfordCoreNLP('10.4.0.15:nlp9000')
nlp_wrapper = StanfordCoreNLP('http://localhost:9000')

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
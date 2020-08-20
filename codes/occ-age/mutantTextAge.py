from textAnnotator import annotatedText as at 
import spacy


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
    nlp = spacy.load('en_core_web_lg')
    mutantSentenceIndex = 999
    isNameFound = False
    sentenceCounter = checkpoint
    name = None
    for i in range(checkpoint, len(sentenceList)):
        text = at(sentenceList[i])
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
    nlp = spacy.load('en_core_web_lg')
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

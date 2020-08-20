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
    nlp = spacy.load('en_core_web_lg')
    mutantSentenceIndex = 999
    isOccupationFound = False
    sentenceCounter = checkpoint
    occupation = None
    mutantSentence = None
    for i in range(checkpoint, len(sentenceList)):
        text = at(sentenceList[i])
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
    return mutantSentence, mutantSentenceIndex, occupation, isOccupationFound, checkedOcc, checkpoint, isLast


def generateMutantOcc(mutantSentence, occupation):
    # funtion to generate mutant sentece
    # input:
        # mutantSentence -> spacy doc
        # occupation -> spacy token
    # output:
        # mutant -> spacy doc
        
    nlp = spacy.load('en_core_web_sm')
    mutant = ''
    for token in mutantSentence:
        if token.text == occupation.text:
            mutant = mutant + ' ' + ''.join("@OCCUPATION")
        else:
            mutant = mutant + ' ' + ''.join(token.text)
    
    # mutant_text = spacy_fulltext.text[0:token_count] + ' @OCCUPATION ' + spacy_fulltext.text[token_count + (len(occupation.text))+2:]
    return (nlp(mutant.strip()))

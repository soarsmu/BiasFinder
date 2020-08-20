import spacy

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
        
    nlp = spacy.load('en_core_web_lg')
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

    nlp = spacy.load('en_core_web_lg')

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

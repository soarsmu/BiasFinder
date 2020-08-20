from textCleaner import symbol
from textCleaner import convertParticles

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
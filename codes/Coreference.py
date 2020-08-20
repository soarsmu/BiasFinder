from CustomToken import CustomToken as Token

# Reference is a class to save Reference data
# e.g. La Marquesa : [La Marquesa, her]
class Coreference:
    
    main_reference = ""
    references = []
    token_references = []
        
    def __init__(self, name, references):
        self.main_reference = str(name)
        self.references = []
        self.token_references = []
        for token in references :
            self.references.append(token.text)
            self.token_references.append(Token(token.text, token.start_char, token.end_char))
            
    def __str__(self) :
        return self.main_reference + ": " + str(self.references)
    
    def __repr__(self) :
        return self.main_reference + ": " + str(self.references)
    
    def getMainReference(self):
        return self.main_reference
    
    def getReferences(self):
        return self.references
    
    def getTokenReferences(self):
        return self.token_references
    
    # is having male subject
    def isHavingMalePersonReference(self):
        if "He" in self.references :
            return True
        elif "he" in self.references :
            return True
        else :
            return False

    # is having female subject
    def isHavingFemalePersonReference(self):
        if "She" in self.references :
            return True
        elif "she" in self.references :
            return True
        else :
            return False
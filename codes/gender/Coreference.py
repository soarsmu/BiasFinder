from Position import Position
from Phrase import Phrase

# Reference is a class to save Reference data
# e.g. La Marquesa : [La Marquesa, her]
class Coreference:
    
    main_reference = ""
    references = []
    position_references = []
        
    def __init__(self, main, references):
        self.main_reference = str(main)
        self.references = []
        self.position_references = []
        self.gender = ""
        for token in references :
            self.references.append(Phrase(token.text))
            self.position_references.append(Position(token.text, token.start_char, token.end_char))
            
    def __str__(self) :
        return self.main_reference + ": " + str(self.references)
    
    def __repr__(self) :
        return self.main_reference + ": " + str(self.references)
    
    def getMainReference(self):
        return self.main_reference
    
    def getReferences(self):
        return self.references
    
    def getPositionReferences(self):
        return self.position_references
    
    def setPlaceholders(self, placeholders) :
        self.placeholders = placeholders
    
    def getPlaceholders(self) :
        return self.placeholders
    
    def getGender(self) :
        return self.gender

    def setGender(self, gender) :
        self.gender = gender

    
    


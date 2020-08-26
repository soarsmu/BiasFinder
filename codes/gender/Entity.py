class Entity:
    word = ""
    start = 0
    end = 0
    ent_type = ""
    def __init__(self, word, start, end, ent_type) :
        self.word = word
        self.start = start
        self.end = end
        self.ent_type = ent_type
    
    def __str__(self) :
        return self.word
    
    def __repr__(self) :
        return self.word
        
    def getWord(self):
        return self.word
    
    def getStart(self):
        return self.start
    
    def getEnd(self):
        return self.end
    
    def getEntityType(self):
        return self.ent_type
    
    def isPerson(self):
        return self.ent_type == "PERSON" and self.word[-2:] != "'s"
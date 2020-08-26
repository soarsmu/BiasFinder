# contain a word and its location inside the sentence
# The location is indicated by start char and end char
class Position: 
    word = ""
    start = 0
    end = 0
    
    def __init__(self, word, start, end) :
        self.word = word
        self.start = start
        self.end = end
        
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
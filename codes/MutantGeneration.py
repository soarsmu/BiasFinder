import pandas as pd

from Entity import Entity
from CustomToken import CustomToken as Token
from Coreference import Coreference
from utils import nlp

# masculine pronoun
masculine_pronoun = ["he", "him", "his", "himself", "He", "Him", "His", "Himself"]

# feminine prononun
feminine_pronoun = ["she", "her", "her", "herself", "She","Her", "Her", "Herself"]

# gender flipper
masculineToFeminine = {}
feminineToMasculine = {}

for _m, _f in zip(masculine_pronoun, feminine_pronoun) :
    masculineToFeminine[_m] = _f
    feminineToMasculine[_f] = _m

# gender associated word
gaw = pd.read_csv("../data/gender_associated_word/masculine-feminine-person.txt")
    
# gender salutation word
male_salutation = ["Mr", "Mr.", "Mr.", "Mister", "Sir"]
female_salutation = ["Ms", "Ms.", "Mrs.", "Miss", "Madam"]

maleToFemaleSalutation = {}
femaleToMaleSalutation = {}
for _m, _f in zip(male_salutation, female_salutation) :
    maleToFemaleSalutation[_m] = _f
    femaleToMaleSalutation[_f] = _m
    

class MutantGeneration:
    original = ""
    resolved = ""
    coreferences = []
    person_entities = []
    is_having_one_person_reference = False
    is_male = False
    chunks = []
    person_reference = None
    person_name = None
    main_placeholder = None
    
    def __init__(self, text):
        
        self.original = str(text)
        doc = nlp(text)
        
        self.person_entities = self.getPersonEntities(doc.ents)
        
        self.resolved = str(doc._.coref_resolved)

        self.coreferences = []
        
        for r in doc._.coref_clusters :
            self.coreferences.append(Coreference(r.main, r.mentions))
        
        self.is_having_one_person_reference, self.person_reference, self.is_male = self.checkIsOnePersonReference()
        
        if self.is_having_one_person_reference :
            self.chunks = self.generateChunkFromCoreference()
        
            
    def getOriginal(self):
        return self.original
    
    def getResolved(self):
        return self.resolved
    
    def getCoreferences(self):
        return self.coreferences
    
    def getGender(self):
        if self.is_male :
            return "male"
        return "female"
    
    def getPersonEntities(self, ents) :
        entities = set()
        for ent in ents :
            e = Entity(ent.text, ent.start_char, ent.end_char, ent.label_)
            if e.isPerson() :
                entities.add(e.getWord())
        return list(entities)
    
    def isHavingOnePersonReference(self) :
        return self.is_having_one_person_reference        
        
    def checkIsOnePersonReference(self) :           
        s = 0
        person_reference = None
        for r in self.coreferences :
            if r.isHavingMalePersonReference() :
                s += 1
                person_reference = r
                is_male = True
            
            if r.isHavingFemalePersonReference() :
                s += 1
                person_reference = r
                is_male = False
                
        if s == 1 :
            # check if it's only prononun there
            is_only_pronoun = True
            print(person_reference.getReferences())
            for r in person_reference.getReferences() :
                if r not in masculine_pronoun and r not in feminine_pronoun :
                    is_only_pronoun = False

            if is_only_pronoun :
                return False, None, None 
            
            return True, person_reference, is_male
        else :
            return False, None, None
    
    def getPersonReference(self):
        return self.person_reference
    
    def generateChunkFromCoreference(self) :
        chunks = []
        trefs = self.person_reference.getTokenReferences()
        lb = 0 # lower bound
        ub = 0 # upper bound
        for i in range(len(trefs)) :
            if i == 0 :
                ub = trefs[i].start
                _chunk = self.original[lb:ub]
                if _chunk == "" :
                    chunks.append(" ")
                else :
                    chunks.append(_chunk)
            else :
                lb = trefs[i-1].end
                ub = trefs[i].start
                _chunk = self.original[lb:ub]
                if _chunk == "" :
                    chunks.append(" ")
                else :
                    chunks.append(_chunk)
                
            if i == len(trefs)-1 :
                lb = trefs[-1].end
                chunks.append(self.original[lb:])
        
        return chunks
    
    def isAPersonName(self, text) :
        if len(self.person_reference.getMainReference()) > 2 :
            if self.person_reference.getMainReference()[-2:] == "'s" :  # remove the main reference if it's contain an apostrophe in the last (this cauased by neural coref library)
                return False
        return text in self.person_entities 
    
    def isTheMainReferenceAPersonName(self) :        
        return self.isAPersonName(self.person_reference.getMainReference())
    
    def isContainAPersonNameAndItIsTheRoot(self, text) :
        doc_text = nlp(text)

        for token in doc_text:
#             print(token.text, token.pos_, token.dep_)
            if token.text in self.person_entities and token.dep_ == "ROOT":        
                return True
        return False
    
    def markGenderSalutationWord(self, text):
        doc = nlp(text)
        
        salutations = []
        if self.is_male :
            salutation = male_salutation
        else :
            salutation = female_salutation
        
        placeholder = []
        for token in doc:
            if token.text in salutation :
                placeholder.append("<sltn-" + token.text + ">")
            elif token.text in self.person_entities :
                if token.dep_ == "ROOT":
                    placeholder.append("<name>")
            else :
                placeholder.append(token.text)

        return " ".join(placeholder)
        
    def isThePersonNameSubstringOfTheMainReference(self) :
        return self.isContainAPersonNameAndItIsTheRoot(self.person_reference.getMainReference())
    
    def isTheRootOfTheTextAGenderAssociatedWord(self, text):
        doc_text = nlp(text)
        
        placeholder = []
        
        check = False

        main_token = None
        for token in doc_text:
#             print(token.text, token.pos_, token.dep_)
            if token.pos_ == "NOUN" and token.dep_ == "ROOT" :
                main_token = token.text
                if self.is_male :
                    if main_token in gaw["masculine"].values :
                        check = True
                else :
                    if main_token in gaw["feminine"].values :
                        check = True
                if check :
                    placeholder.append("<gaw>")
            else :
                placeholder.append(token.text)
        
        if check :
            self.main_placeholder = " ".join(placeholder)

        return check
        
    
    def isTheMainReferenceAGenderAssociatedWord(self) :
        return self.isTheRootOfTheTextAGenderAssociatedWord(self.person_reference.getMainReference())
    
    def isTheMainReferenceAPersonPronoun(self):
        return self.person_reference.getMainReference() in (masculine_pronoun + feminine_pronoun)
    
    def isTheReferencesContainNonPronoun(self) :
        for word in self.person_reference.getReferences() :
            if self.isAPersonName(word):
                self.main_placeholder = "<name>"
                return True
            if self.isContainAPersonNameAndItIsTheRoot(word) :
                self.main_placeholder = self.markGenderSalutationWord(word)
                return True
            if self.isTheRootOfTheTextAGenderAssociatedWord(word) :
                return True
        return False
    

            
    def generateTemplate(self) :
        
        main_placeholder = None
        
        if self.isHavingOnePersonReference() :
            if self.isTheMainReferenceAPersonName() :
                print("The Main Reference is a Person Name")
                main_placeholder = "<name>"
            elif self.isThePersonNameSubstringOfTheMainReference() :
                print("Person Name Substring of The Main Reference")
                main_placeholder = self.markGenderSalutationWord(self.person_reference.getMainReference())
            elif self.isTheMainReferenceAGenderAssociatedWord() :
                print("The Main Reference is a Gender Associated Word")
                main_placeholder = self.main_placeholder
            elif self.isTheMainReferenceAPersonPronoun() :
                print("The Main Reference is a Person Pronoun")
                if self.isTheReferencesContainNonPronoun() :
                    print("The References Contain Non Prononun")
                    main_placeholder = self.main_placeholder
                else :
                    print("The References is Not Replaceable")
                    
            print(main_placeholder)
        
            t = [self.chunks[0]]
            i = 1
            if self.is_male :
                for r in self.person_reference.getTokenReferences() :
                    if r.word in masculine_pronoun :
                        t.append("<pro-" + r.word + ">")
                    else :
                        t.append(main_placeholder)

                    t.append(self.chunks[i])
                    i += 1

            self.template = " ".join(t).strip()
            print(self.template)
        
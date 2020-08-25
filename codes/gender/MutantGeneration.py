import re
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

NAME = "name"
PRONOUN = "pro"
SALUTATION = "sltn"
GAW = "gaw"

for _m, _f in zip(masculine_pronoun, feminine_pronoun) :
#     masculineToFeminine["<" + PRONOUN + "-" + _m + ">"] = _f
#     feminineToMasculine["<" + PRONOUN + "-" + _f + ">"] = _m
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
#     maleToFemaleSalutation["<" + SALUTATION + "-" + _m + ">"] = _f
#     femaleToMaleSalutation["<" + SALUTATION + "-" + _f + ">"] = _m
    maleToFemaleSalutation[_m] = _f
    femaleToMaleSalutation[_f] = _m

# mnames = ["Alonzo", "Adam", "Alphonse", "Alan", "Darnell", "Andrew", "Jamel", "Frank", "Jerome", "Harry", "Lamar", "Jack", "Leroy", "Josh", "Malik", "Justin", "Terrence", "Roger", "Torrance", "Ryan"]
# fnames = ["Ebony", "Amanda", "Jasmine", "Betsy", "Lakisha", "Courtney", "Latisha", "Ellen", "Latoya", "Heather", "Nichelle", "Katie", "Shaniqua", "Kristin", "Shereen", "Melanie", "Tanisha", "Nancy", "Tia", "Stephanie"]
# mcountries = ["African American"] * 10
# mcountries.extend(["European American"] * 10)
# fcountries = mcountries.copy()


# load name from gender computer
# gc = pd.read_csv("../data/gc_name/data.csv")
# gcm = gc[gc["Gender"] == "male"]
# gcf = gc[gc["Gender"] == "female"]
# # names from GC
# # gcm = gcm[:2]
# # gcf = gcf[:2]
# mnames = gcm["Name"].tolist()
# mcountries = gcm["Country"].tolist()
# fnames = gcf["Name"].tolist()
# fcountries = gcf["Country"].tolist()


# small name for debugging
mnames = ["Alonzo", "Adam"] 
fnames = ["Ebony", "Amanda"]
mcountries = ["Trial", "Trial"]
fcountries = ["Trial", "Trial"]


countries = mcountries.copy()
countries.extend(fcountries)

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
    template = None
    
    def __init__(self, text):
        
        self.original = str(text)
        self.docs = nlp(text)
        
        self.person_entities = self.getPersonEntities()
        
        self.resolved = str(self.docs._.coref_resolved)

        self.coreferences = []
        
        for r in self.docs._.coref_clusters :
            self.coreferences.append(Coreference(r.main, r.mentions))
        
        self.is_having_one_person_reference, self.person_reference, self.is_male = self.checkIsOnePersonReference()
        
        if self.is_having_one_person_reference :
            self.chunks = self.generateChunkFromCoreference()
            self.generateTemplate()
        
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
    
    def getPersonEntities(self) :
        entities = set()
        for ent in self.docs.ents :
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
#             print(person_reference.getReferences())
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
        is_any_salutation = False
        for token in doc:
            if token.text in salutation :
                placeholder.append("<" + SALUTATION + "-" + token.text + ">")
                is_any_salutation = True
            elif token.text in self.person_entities :
                if token.dep_ == "ROOT":
                    placeholder.append("<name>")
            else :
                placeholder.append(token.text)

        return " ".join(placeholder), is_any_salutation
        
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
                    placeholder.append("<" + GAW + ">")
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
                self.main_placeholder = "<" + NAME + ">"
                self.main_placeholder_type = NAME
                return True
            if self.isContainAPersonNameAndItIsTheRoot(word) :
                self.main_placeholder, is_any_salutation = self.markGenderSalutationWord(word)
                if is_any_salutation :
                    self.main_placeholder_type = SALUTATION
                    return True
            if self.isTheRootOfTheTextAGenderAssociatedWord(word) :
                self.main_placeholder_type = GAW
                return True
        return False
    

            
    def generateTemplate(self) :
        
        main_placeholder = None
        main_placeholder_type = None
        self.is_person_pronoun = False
        
        if self.isHavingOnePersonReference() :
            is_replacable = False
            if self.isTheMainReferenceAPersonName() :
#                 print("The Main Reference is a Person Name")
                main_placeholder = "<" + NAME + ">"
                main_placeholder_type = NAME
                is_replacable = True
            elif self.isThePersonNameSubstringOfTheMainReference() :
#                 print("Person Name Substring of The Main Reference")
                main_placeholder, is_any_salutation = self.markGenderSalutationWord(self.person_reference.getMainReference())
                if is_any_salutation :
                    main_placeholder_type = SALUTATION
                    is_replacable = True
                else :
                    main_placeholder_type = NAME
                    is_replacable = False
            elif self.isTheMainReferenceAGenderAssociatedWord() :
#                 print("The Main Reference is a Gender Associated Word")
                main_placeholder = self.main_placeholder
                main_placeholder_type = GAW
                is_replacable = True
            elif self.isTheMainReferenceAPersonPronoun() :
#                 print("The Main Reference is a Person Pronoun")
                if self.isTheReferencesContainNonPronoun() :
#                     print("The References Contain Non Prononun")
                    main_placeholder = self.main_placeholder
                    main_placeholder_type = self.main_placeholder_type
                    is_replacable = True
                    self.is_person_pronoun = True
#                 else :
#                     print("The References is Not Replaceable")
                            
            if is_replacable :
                t = [self.chunks[0]]
                i = 1
                if self.is_male :
                    for r in self.person_reference.getTokenReferences() :
                        if r.word in masculine_pronoun :
                            t.append("<" + PRONOUN + "-" + r.word + ">")
                        else :
                            t.append(main_placeholder)

                        t.append(self.chunks[i])
                        i += 1    
                else :
                    for r in self.person_reference.getTokenReferences() :
                        if r.word in feminine_pronoun :
                            t.append("<" + PRONOUN + "-" + r.word + ">")
                        else :
                            t.append(main_placeholder)

                        t.append(self.chunks[i])
                        i += 1

                        
                template = " ".join(t).strip()
                self.template = re.sub(' +', ' ', template)
                self.main_placeholder_type = main_placeholder_type
            
#                 print(main_placeholder)
#                 print("CHUNK: " + str(self.chunks))
#                 print("TEMPLATE: " + self.template)

            
    def replacePronounTemplateIntoMalePronoun(self, template) :
        if self.is_male :
            for token in masculine_pronoun :
                template = template.replace("<" + PRONOUN + "-" + token + ">", token)
            return template
        else :
            for token in feminine_pronoun :
                template = template.replace("<" + PRONOUN + "-" + token + ">", feminineToMasculine[token])
            return template

    
    def generateMutantUsingName(self, template, names) :
        mutants = [] 
        for name in names :
            _template = template.replace("<" + NAME + ">", name)
            mutants.append(_template)
        return mutants
    
    def generateMaleMutantUsingNameAndSalutation(self, template, names) :
        if self.is_male :
            for m in male_salutation:
                template = template.replace("<" + SALUTATION + "-" + m + ">", m)
        else :
            for f in female_salutation:
                template = template.replace("<" + SALUTATION + "-" + f + ">", femaleToMaleSalutation[f])
        return self.generateMutantUsingName(template, names)
    
    def generateMaleMutantUsingGenderAssociatedWord(self, template, gaw) :
        mutants = []
        for m in gaw["masculine"] :
            mutants.append(template.replace("<" + GAW + ">", m))
        return mutants
    
    
    def generateMaleMutant(self) :
        mutants = []
        identifiers = []
        types = []
        template = self.replacePronounTemplateIntoMalePronoun(self.template)
        if self.main_placeholder_type == NAME :
            mutants = self.generateMutantUsingName(template, mnames)
            identifiers = mnames.copy() 
            types = [NAME] * len(mnames)
        elif self.main_placeholder_type == SALUTATION :
            mutants = self.generateMaleMutantUsingNameAndSalutation(template, mnames)
            identifiers = mnames.copy()
            types = [SALUTATION] * len(mnames)
        elif self.main_placeholder_type == GAW :
            mutants = self.generateMaleMutantUsingGenderAssociatedWord(template, gaw)
            identifiers = gaw["masculine"].tolist()
            types = [GAW] * len(gaw["masculine"])
                
        return mutants, identifiers, types
    

    def replacePronounTemplateIntoFemalePronoun(self, template) :
        if self.is_male :
            for token in masculine_pronoun :
                template = template.replace("<" + PRONOUN + "-" + token + ">", masculineToFeminine[token])
            return template
        else :
            for token in feminine_pronoun :
                template = template.replace("<" + PRONOUN + "-" + token + ">", token)
            return template
        
    def generateFemaleMutantUsingNameAndSalutation(self, template, names) :
        if self.is_male :
            for m in male_salutation:
                template = template.replace("<" + SALUTATION + "-" + m + ">", maleToFemaleSalutation[m])
        else :
            for f in female_salutation:
                template = template.replace("<" + SALUTATION + "-" + f + ">", f)
        return self.generateMutantUsingName(template, names)
    
    def generateFemaleMutantUsingGenderAssociatedWord(self, template, gaw) :
        mutants = []
        for m in gaw["feminine"] :
            mutants.append(template.replace("<" + GAW + ">", m))
        return mutants

    def generateFemaleMutant(self) :
        mutants = []
        identifiers = []
        types = []
        template = self.replacePronounTemplateIntoFemalePronoun(self.template)
        if self.main_placeholder_type == NAME :
            mutants = self.generateMutantUsingName(template, fnames)
            identifiers = fnames.copy() 
            types = [NAME] * len(fnames)
        elif self.main_placeholder_type == SALUTATION :
            mutants = self.generateFemaleMutantUsingNameAndSalutation(template, fnames)
            identifiers = fnames.copy() 
            types = [SALUTATION] * len(fnames)
        elif self.main_placeholder_type == GAW :
            mutants = self.generateFemaleMutantUsingGenderAssociatedWord(template, gaw)
            identifiers = gaw["feminine"].tolist()
            types = [GAW] * len(gaw["feminine"])
            
        return mutants, identifiers, types
    
    def generateMutant(self) :
        if self.template :

            male_mutants, male_identifiers, male_types = self.generateMaleMutant()
            female_mutants, female_identifiers, female_types = self.generateFemaleMutant()

            genders = ["male"] * len(male_mutants)
            genders.extend(["female"] * len(female_mutants))

            male_mutants.extend(female_mutants)
            male_identifiers.extend(female_identifiers)
            male_types.extend(female_types)

            countries = []

            if self.main_placeholder_type == NAME or self.main_placeholder_type == SALUTATION :
                countries = mcountries.copy()
                countries.extend(fcountries)
            elif self.main_placeholder_type == GAW :
                countries = [None] * len(male_types)
                
            if self.is_person_pronoun :
                for i in range(len(male_types)) :
                    male_types[i] = "person-pronoun-" + male_types[i]

            return male_mutants, [self.template] * len(male_mutants), male_identifiers, male_types, genders, countries
        
        return [], [], [], [], [], []
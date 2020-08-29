import re
import pandas as pd

from utils import NAME, PRONOUN, GAW, SALUTATION

from Entity import Entity
from Phrase import Phrase
from Coreference import Coreference
from utils import nlp
from utils import tag, preprocessText
from utils import getPronounPlaceholders
from utils import masculineToFemininePronoun
from utils import feminineToMasculinePronoun
from utils import masculineToFeminineSalutation
from utils import feminineToMasculineSalutation
from utils import getMaleNamesFromGenderComputer, getFemaleNamesFromGenderComputer
from utils import getMasculineGenderAssociatedWord, getFeminineGenderAssociatedWord 

class MutantGeneration:
    original = ""
    coreferences = []
    person_entities = []
    chunks = []
    template = ""
    examples = []
    mutants = []
    templates = []
    genders = []
    valid_coreferences = []
    
    def __init__(self, text):
        
        self.original = str(text)
        self.docs = nlp(text)
        
        self.person_entities = self.getPersonEntities()

        # self.coreferences = []
        self.person_coreferences = []
        self.person_coreferences = self.getPersonCoreferences()
        if len(self.person_coreferences) == 1 :
            coref = self.person_coreferences[0]
            if self.isValid(coref) :
                template = self.generateTemplate(coref)
                self.templates, self.mutants, self.genders = self.generateMutant(coref, template)
            

    def getPersonCoreferences(self) :
    
        coreferences = []
        for r in self.docs._.coref_clusters :
            coref = Coreference(r.main, r.mentions)
            if self.isPersonCoref(coref) : # only take valid coreference
                coreferences.append(coref)
        
        return coreferences

    def isValid(self, coref):
        placeholders = []
#         print("SELECTED COREF: " + str(coref.getReferences()))
        gender = ""
        for phrase in coref.getReferences() :
            if phrase.isGenderPronoun():
                if gender == "" :
                    gender = phrase.getGender()
                elif gender != phrase.getGender() : ## there is 2 gender pronoun detected
                    return False
                coref.setGender(phrase.getGender())
                id = phrase.getPhrase()
                placeholders.append(tag(PRONOUN + "-" + id))
            elif self.isPersonName(phrase.getPhrase()):
                placeholders.append(tag(NAME))
            elif phrase.isContainGenderAssociatedWord() :
                gaw = phrase.getGenderAssociatedWord()
                placeholder = phrase.getPhrase().replace(gaw, tag(GAW))
                placeholders.append(placeholder)
            else :
                return False

        if coref.getGender() == "" : # no pronoun found
            return False
        
        coref.setPlaceholders(placeholders)

        ## replace <name><name> into <name>
        ## how if all pronoun -> pass
        return True
            
    def isPersonCoref(self, coref):
        placeholders = []
#         print("COREF: " + str(coref.getReferences()))
        gender = ""
        for phrase in coref.getReferences() :
            if phrase.isGenderPronoun():
                return True
#                 if gender == "" :
#                     gender = phrase.getGender()
#                 elif gender != phrase.getGender() : ## there is 2 gender pronoun detected
#                     return False
#                 coref.setGender(phrase.getGender())
#                 id = phrase.getPhrase()
#                 placeholders.append(tag(PRONOUN + "-" + id))
            elif self.isPersonName(phrase.getPhrase()):
                return True
#                 placeholders.append(tag(NAME))
            elif phrase.isContainGenderAssociatedWord() :
                return True
#                 gaw = phrase.getGenderAssociatedWord()
#                 placeholder = phrase.getPhrase().replace(gaw, tag(GAW))
#                 placeholders.append(placeholder)

#         if coref.getGender() == "" : # no pronoun found
#             return False
        
#         coref.setPlaceholders(placeholders)

        ## replace <name><name> into <name>
        ## how if all pronoun -> pass
        return False

    
    def getOriginal(self):
        return self.original

    def getCoreferences(self):
        return self.coreferences
    
    def getTemplates(self):
        return self.templates
    
    def getGenders(self):
        return self.genders
    
    def getMutants(self) :
        return self.mutants
    
    def isPersonName(self, text) :
        return text in self.person_entities 

    def isContainPersonName(self, phrase) :
        tokens = phrase.getTokens()
        for token in tokens:
            if token.text in self.person_entities :
                return True
        return False

    def getPersonEntities(self) :
        entities = set()
        for ent in self.docs.ents :
            e = Entity(ent.text, ent.start_char, ent.end_char, ent.label_)
            if e.isPerson() :
                entities.add(e.getWord())
#                 print("PERSON: ", e.getWord())
#         print()
        return list(entities)
            
    def generateTemplate(self, coref) :
        chunks = self.generateChunk(coref)
        placeholders = coref.getPlaceholders()
        
        tokens = [chunks[0]]
        i = 1
        
        for placeholder in placeholders :
            tokens.append(placeholder)
            tokens.append(chunks[i])
            i += 1  

        template = " ".join(tokens).strip()
        template = re.sub(' +', ' ', template)

#         print(self.original)
#         print(chunks)
#         print(placeholders)
#         print(template)
        self.template = str(template)
        return template
    
    def getTemplate(self):
        return self.template
    
    def getMutants(self):
        return self.mutants
        
    def getMutantExamples(self):
        return self.examples

        
    def generateChunk(self, coref) :
        chunks = []
        refs = coref.getPositionReferences()
        lb = 0 # lower bound
        ub = 0 # upper bound
        for i in range(len(refs)) :
            if i == 0 :
                ub = refs[i].start
                chunk = self.original[lb:ub]
                chunks.append(chunk)
            else :
                lb = refs[i-1].end
                ub = refs[i].start
                chunk = self.original[lb:ub]
                chunks.append(chunk)
            if i == len(refs)-1 :
                lb = refs[-1].end
                chunks.append(self.original[lb:])
        return chunks
    
    
    def generateMutant(self, coref, template) :
        templates = []
        mutants = []
        genders = []
        
        self.examples = []
        placeholders = coref.getPlaceholders()
        used_placeholders = set(placeholders)
        gender = coref.getGender()
            
        male_mutants = self.generateMaleMutant(template, used_placeholders, gender)
        female_mutants = self.generateFemaleMutant(template, used_placeholders, gender)
        
#         print("LEN M: ", len(male_mutants))
#         print("LEN F: ", len(female_mutants))

        
        if len(male_mutants) == len(female_mutants) and len(male_mutants) > 0 and len(female_mutants) > 0 :
            templates = [self.template] * 2 * len(male_mutants)
            genders = ["male"] * len(male_mutants)
            genders.extend(["female"] * len(female_mutants))
            mutants.extend(male_mutants)
            mutants.extend(female_mutants)
            
            self.examples.append(male_mutants[0])
            self.examples.append(female_mutants[0])

        return templates, mutants, genders

    
    
    def generateMaleMutant(self, template, placeholders, gender) :
        
        pronoun_placeholders = getPronounPlaceholders(placeholders)
                
        template = self.replaceGenderPronounPlaceholderIntoMale(template, pronoun_placeholders, gender)

#         print("TEMPLATE: " + template)

        templates = [template]
        
        non_pronoun_placeholders = placeholders.difference(pronoun_placeholders)
          
        for placeholder in non_pronoun_placeholders :
#             print("PLACEHOLDER: " + placeholder)
            if placeholder == tag(NAME) :
#                 print("XXXX")
                templates = self.replaceNamePlaceholder(templates, getMaleNamesFromGenderComputer())
            elif tag(GAW) in placeholder :
                templates = self.replaceGenderAssociatedWordPlaceholder(templates, getMasculineGenderAssociatedWord())
#             elif "<" + SALUTATION in placeholder :
#                 templates = self.replaceSalutationPlaceholderIntoMale(templates, placeholder, gender)
#                 templates = self.replaceNamePlaceholder(templates, getMaleNamesFromGenderComputer())
            else :
                raise Exception

#         print(templates)
        return templates
                
    def generateFemaleMutant(self, template, placeholders, gender) :
        
        pronoun_placeholders = getPronounPlaceholders(placeholders)
                
        template = self.replaceGenderPronounPlaceholderIntoFemale(template, pronoun_placeholders, gender)

        templates = [template]
        
        non_pronoun_placeholders = placeholders.difference(pronoun_placeholders)
          
        for placeholder in non_pronoun_placeholders :
#             print("PLACEHOLDER: " + placeholder)
            if placeholder == tag(NAME) :
#                 print("XXXX")
                templates = self.replaceNamePlaceholder(templates, getFemaleNamesFromGenderComputer())
            elif tag(GAW) in placeholder :
                templates = self.replaceGenderAssociatedWordPlaceholder(templates, getFeminineGenderAssociatedWord())
#             elif "<" + SALUTATION in placeholder :
#                 templates = self.replaceSalutationPlaceholderIntoFemale(templates, placeholder, gender)
#                 templates = self.replaceNamePlaceholder(templates, getFemaleNamesFromGenderComputer())
            else :
                raise Exception

        return templates
    
    
    def replaceGenderPronounPlaceholderIntoMale(self, template, placeholders, gender) :
        if gender == "male" :
            for placeholder in placeholders :
                token = placeholder[5:-1] #get token
                template = template.replace(placeholder, token)
            return template
        else :
            for placeholder in placeholders :
                token = placeholder[5:-1] #get token
#                 print("TOKEN: " + token)
#                 print("TEMPLATE: " + template)
                template = template.replace(placeholder, feminineToMasculinePronoun(token))
            return template
        
    def replaceGenderPronounPlaceholderIntoFemale(self, template, placeholders, gender) :
        if gender == "male" :
            for placeholder in placeholders :
                token = placeholder[5:-1] #get pronoun
                template = template.replace(placeholder, masculineToFemininePronoun(token))
            return template
        else :
            for placeholder in placeholders :
                token = placeholder[5:-1] #get token
                template = template.replace(placeholder, token)
            return template

    
    def replaceNamePlaceholder(self, src_templates, names) :
        templates = [] 
#         print("XXXX")
        for template in src_templates :
            if tag(NAME) in template :
                for name in names :
#                     print("XXXXXXX")
                    _template = template.replace(tag(NAME), name.title())
                    templates.append(_template)
            else :
                templates.append(template)
        return templates
    
    def replaceSalutationPlaceholderIntoMale(self, src_templates, placeholder, gender) :
        placeholder = placeholder[:-7]
        token = placeholder[6:-1] #get token
#         print("TOKEN: " + token)
        templates = []
        for template in src_templates :
            if placeholder in template :
                if gender == "male" :
                    template = template.replace(placeholder, token)
                    templates.append(template)
                else :
                    template = template.replace(placeholder, feminineToMasculineSalutation(token))
                    templates.append(template)
            else :
                templates.append(template)
        return templates
    
    def replaceSalutationPlaceholderIntoFemale(self, src_templates, placeholder, gender) :
        placeholder = placeholder[:-7]
        token = placeholder[6:-1] #get token
#         print("TOKEN: " + token)
        templates = []
        for template in src_templates :
            if placeholder in template :
                if gender == "male" :
                    template = template.replace(placeholder, masculineToFeminineSalutation(token))
                    templates.append(template)
                else :
                    template = template.replace(placeholder, token)
                    templates.append(template)
            else :
                templates.append(template)
        return templates
    
    def replaceGenderAssociatedWordPlaceholder(self, src_templates, gaw) :
        templates = []
        for template in src_templates :
            if tag(GAW) in template :
                for word in gaw :
                    templates.append(template.replace(tag(GAW), word))
            else :
                templates.append(template)
        return templates
                
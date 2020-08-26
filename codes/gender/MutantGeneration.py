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
    
    def __init__(self, text):
        
        self.original = str(text)
        self.docs = nlp(text)
        
        self.person_entities = self.getPersonEntities()

        self.coreferences = []
        
        for r in self.docs._.coref_clusters :
            coref = Coreference(r.main, r.mentions)
            if self.isValid(coref) : # only take valid coreference
#                 self.coreferences.append(c)
                template = self.generateTemplate(coref)
                mutants = self.generateMutant(coref, template)
        
        
    def getOriginal(self):
        return self.original
    
    def getCoreferences(self):
        return self.coreferences
    
    def isValid(self, coref):
        placeholders = []
        for phrase in coref.getReferences() :
            if phrase.isGenderPronoun():
                coref.setGender(phrase.getGender())
                id = phrase.getPhrase()
                placeholders.append(tag(PRONOUN + "-" + id))
            elif self.isPersonName(phrase.getPhrase()):
                placeholders.append(tag(NAME))
            elif phrase.isHasSalutation() and self.isContainPersonName(phrase):
                id = phrase.getSalutation()
                placeholder = tag(SALUTATION + "-" + id) + " " + tag(NAME)
                placeholders.append(placeholder)
            elif phrase.isContainGenderAssociatedWord() :
                gaw = phrase.getGenderAssociatedWord()
                placeholder = phrase.getPhrase().replace(gaw, tag(GAW))
                placeholders.append(placeholder)
            else :
                return False

        coref.setPlaceholders(placeholders)

        ## replace <name><name> into <name>
        ## how if all pronoun
        return True

        
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
        return template
        
    
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
        mutants = []
        placeholders = coref.getPlaceholders()
        used_placeholders = set(placeholders)
        gender = coref.getGender()
            
        self.generateMaleMutant(template, used_placeholders, gender)
        self.generateFemaleMutant(template, used_placeholders, gender)

        return mutants
    
    
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
            elif "<" + SALUTATION in placeholder :
                templates = self.replaceSalutationPlaceholderIntoMale(templates, placeholder, gender)
                templates = self.replaceNamePlaceholder(templates, getMaleNamesFromGenderComputer())
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
            elif "<" + SALUTATION in placeholder :
                templates = self.replaceSalutationPlaceholderIntoFemale(templates, placeholder, gender)
                templates = self.replaceNamePlaceholder(templates, getFemaleNamesFromGenderComputer())
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
            for name in names :
#                 print("XXXXXXX")
                _template = template.replace(tag(NAME), name)
                templates.append(_template)
        return templates
    
    def replaceSalutationPlaceholderIntoMale(self, src_templates, placeholder, gender) :
        placeholder = placeholder[:-7]
        token = placeholder[6:-1] #get token
#         print("TOKEN: " + token)
        templates = []
        for template in src_templates :
            if gender == "male" :
                template = template.replace(placeholder, token)
                templates.append(template)
            else :
                template = template.replace(placeholder, feminineToMasculineSalutation(token))
                templates.append(template)
        return templates
    
    def replaceSalutationPlaceholderIntoMale(self, src_templates, placeholder, gender) :
        placeholder = placeholder[:-7]
        token = placeholder[6:-1] #get token
#         print("TOKEN: " + token)
        templates = []
        for template in src_templates :
            if gender == "male" :
                template = template.replace(placeholder, masculineToFeminineSalutation(token))
                templates.append(template)
            else :
                template = template.replace(placeholder, token)
                templates.append(template)
        return templates
    
    def replaceGenderAssociatedWordPlaceholder(self, src_templates, gaw) :
        templates = []
        for template in src_templates :
            for word in gaw :
                templates.append(template.replace(tag(GAW), word))
        return templates
                
    
    
# there is a person name
text = "When Nathaniel Kahn embarked into this voyage, he hardly knew who his father really was. By the end of the film, he found him and comes to terms with the strange life he lived as a child."

# It contain a person name and has a salutation
text = "Meek and mild Edward G. Robinson (as Wilbert Winkle) decides to quit his bank job and do what he wants, open a ”fix-it” repair shop behind his house. Mr. Robinson is married, but childless; he has befriended local orphanage resident Ted Donaldson (as Barry)"

# text = "I'm sorry, but \" Star Wars Episode 1 \" did not do any justice to Natalie Portman's talent ( and undeniable cuteness). She was entirely underused as Queen Amidala, and when she was used, her makeup was frighteningly terrible. For \" Anywhere But Here, \" she sheds her godawful makeup and she acts normally. And not only can she act good, she looks good doing it. I'm a bit older than she ( she's only 18), and I have little or no chance of meeting her, but hey, a guy is allowed to dream, right? Even though Susan Sarandon does take a good turn in this movie, the film belongs entirely to Portman. I've been a watcher of Portman's since \" Beautiful Girls \" ( where she was younger, but just as cute). There's big things for her in the future. I can see it."

# text = "In this film I prefer Deacon Frost. He's so sexy! I love his glacial eyes! I like Stephen Dorff and the vampires, so I went to see it. I hope to see a gothic film with him. \" Blade \" it was very \" about the future \". If vampires had been real, I would be turned by Frost!"

text = "Mr. Bean has shaped the face of British TV comedy. He has proved that you do not need wicked words or wit, a massive budget, a great deal of intelligence or even any intelligence to make something brilliant. And Mr. Bean is one of those characters who you just can't forget."

# the gender associated word
text = "Even the manic loony who hangs out with the bad guys in ”Mad Max” is there. That guy from ”Blade Runner” also cops a good billing, although he only turns up at the beginning and the end of the movie."

# the main reference is a pronoun
# text = "This movie is about a man who likes to blow himself up on gas containers. He also loves his mommy. So, to keep the money coming in, he takes his act to Broadway. See! Cody Powers Jarrett blow himself up on his biggest gas container yet!"

# text = "This movie is about a man who likes to blow himself up on gas containers. He also loves his mommy. So, to keep the money coming in, he takes his act to Broadway. SEE! CODY POWERS JARRETT BLOW HIMSELF UP ON HIS BIGGEST GAS CONTAINER YET! TONIGHT! 7.30PM!  However, one day, his mommy dies and Jarrett goes berserk. He kidnaps the audience in the theatre and makes them all stand on top of a huge gas cylinder. Losing control further, he makes them all scream \"MADE IT MA, TOP OF THE CYLINDER!\" in unison. The noise is so deafening that it bursts Jarrets eardrums, causing him to topple from the cylinder into a vat of acid. This Warner Bros. movie is not all it's cracked up to be."

text = preprocessText(text)

MutantGeneration(text)


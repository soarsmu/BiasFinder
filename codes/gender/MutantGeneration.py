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
    
    def __init__(self, text):
        
        self.original = str(text)
        self.docs = nlp(text)
        
        self.person_entities = self.getPersonEntities()

        self.coreferences = []
        
        for r in self.docs._.coref_clusters :
            coref = Coreference(r.main, r.mentions)
            if self.isValid(coref) : # only take valid coreference
                template = self.generateTemplate(coref)
                self.templates, self.mutants, self.genders = self.generateMutant(coref, template)
                break
        
        
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
    
    def isValid(self, coref):
        placeholders = []
#         print("COREF: " + str(coref.getReferences()))
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
            elif phrase.isHasSalutation() and self.isContainPersonName(phrase):
                if gender == "" :
                    gender = phrase.getGender()
                elif gender != phrase.getGender() : ## there is 2 different gender detected
                    return False
                id = phrase.getSalutation()
                placeholder = tag(SALUTATION + "-" + id) + " " + tag(NAME)
                placeholders.append(placeholder)
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
        
        print("LEN M: ", len(male_mutants))
        print("LEN F: ", len(female_mutants))

        
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
                    _template = template.replace(tag(NAME), name)
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

# library problem
text = "Sometimes Hallmark can get it right - like The 10th Kingdom - but many of their fantasy films plod, and this falls into the latter category. The version I saw may have been cut (a demon [?] shown in the trailer and publicity stills didn't appear), but anything that made the movie shorter can only be a blessing. POSSIBLE SPOILERS IF YOU ARE UNFAMILIAR WITH THE ORIGINAL FAIRY TALE: Anyway, the film updates the story to the early part of the 20th Century (?), and makes Gerda and Kay (here called Kai - being a Lexx fan, I kept expecting him to say, `The Dead do not solve puzzles') 18 year olds. Hans Christian Andersen's basic story is followed: the boy gets a shard of ice in his eye, goes bad, is taken off by the Snow Queen to solve a puzzle in her palace and Gerda goes to find him, having various adventures on the way. As the two main characters are older than in the original, a lot of time is spent getting them together and `in love'. Unfortunately, I was never convinced that they were particularly in love, and certainly not enough in love to make sense of Gerda's quest. By the time the main plot kicks in, the movie's pace has slowed to a crawl. Alas, when Gerda begins her search for Kai, it only manages to pick up the pace to a leisurely stroll. There are a few odd additions to the story that seem to go nowhere. At the start of the film the Snow Queen kills Gerda's mother, but no explanation for this is given. A polar bear living in the Snow Queen's palace is more than he seems (though this is possibly because the producers realised that the bear's feelings towards the Snow Queen would be OK in a Fairy Tale, but not in a modern film). Again, this is never explained. Also, hints that the Snow Queen has an erotic desire for Kai are dropped, but never followed through. The script is also full of anachronisms that really jar you out of the `fairy tale' mood. The production looks good, though there is evidence of penny-pinching: the Snow Queen's palace is the hotel where Gerda and Kai lived covered in ice. The three main characters are played with varying degrees of success: Kai comes across as bland as does Gerda initially, but once she sets off to find Kai you warm to her. Bridget Fonda looks great as the Snow Queen, but seems to be in a different movie to everyone else. Ultimately, the film is unsatisfying. It looks good, but drags and lacks magic."

text = "My family watches this movie over and over. Even our 3 year old loves it. I like the \" goodness \" in the movie. Giving the stranger a chance ... showing goodwill to one obviously in need of some unconditional acceptance. The movie gives a feeling of goodwill and victory. One other aspect of the movie that makes it so appealing is the personalities of Velvet's siblings. The bird lover. The bug lover. The boy lover! Very cute and happy movie. There is one thing, however that is irritating about it. That is how Mrs. Brown often makes Mr. Brown look foolish or unwise. She, at times, comes off as a know it all, and he as a dimwit, which he is not. Too bad to put that into such a nice story. Nevertheless, we will continue to enjoy this wonderful, old movie!"

text = "i guess if they are not brother this film will became very common. how long were they can keep this? if we were part, what should they do?so natural feelings, so plain and barren words. But I almost cried last night blood relationship brotherhood love knot film.in another word, the elder brother is very cute.if they are not brothers, they won't have so many forbidden factorsfrom the familysocietyfriendseven hearts of their own at the very beginning. The elder brother is doubtful of whether he is coming out or not at the beginning .maybe the little brother being so long time with his brother and even can't got any praise from his fatherthis made him very upset and even sad, maybe this is a key blasting fuse let him feel there were no one in the world loving him except his beloved brother. and i want to say, this is a so humannatural feeling, there is nothing to be shamed, you may fell in love your motherbrothersister. Just a frail heart looking for backbone to rely on"

text = "A fascinating slice of life documentary about a husband and wife and their marriage told through the eyes of their son. We all like to think that our parents lived happy lives, that their marriages were full of fulfilment, love, and happy memories. Sadly many of us know this not to be the case of their own families and that of their parents. This wonderful little documentary is told through the camera lens and emotional perspective of the son of a family that has just experienced the death of their mother. The son being a documentary film maker has filmed his elder family for many years, for as he states \" posterity \". Three months after the death of his mother his father remarries his long time secretary. The suddenness of this occurrence stuns the family and pushes the son to dig into the past lives of his mother and father. What he reveals is a fascinating look into the lives of two rather ordinary people who like so many of their generation married early for the wrong reasons and found themselves stuck in a family life where they found they just had to \" make do \". A wife who found herself at times bitterly lonely and unloved and a husband who buries himself in his work. She and intellectual at heart, he a much simpler individual who seems to find most of his pleasures in the quiet solitude of work. They are obviously wrong for each other, this much is clear. Yet they stick it out, for what? Well that\'s part of the mystery, they clearly show affection for each other at times if not ever much love. You won\'t find any truly shocking disclosures here, aside from infidelity on both sides, which in good part is what makes this such a gem. You really feel that these could be your own parents if circumstances were different and indeed makes one question the lives of ones own parents."

text = "Larry is a perfect example of the Democratic Party in the United States, of which he is a staunch member. King used to be somewhat fair and unbiased and had a variety of guests on. The Party used to be centrist, too, but that was another era. Now, like, Larry, it\'s Far Left.   At least 90 percent of all the guests on King\'s show in the past year or two are Liberals who sit there and bash President Bush and every Conservative they can think of ..... night after night. Bill Mahar, one of the more viscous ones, isand you can look this upthe most frequent guest in the history of King\'s TV show. You can count on other outspoken Left Wingers to be on King\'s show each week, but don\'t hold your breath waiting for a Conservative. They are few and far between. King was also one of the innovators of the media overkill. That all began with the O.J. Simpson trial. Night after night after night that\'s all you ever saw back in the mid \' 90s. Whatever latest gossip on Anna Nicole Smith, or the Petersen murder case, or Paris Hilton, Britney Spears or some other tabloid subject, you can bet Larry will beat it to death. Sadly, all the other networks do the same thing now. Larry was a leader in that regard. King also has the nerve to sometimes give advice, such as on marriage. I am not kidding; I \' ve heard him say it. The joke is that he has been married and divorced a half dozen times! This man has few scruples, believe me. When it comes to morality, he is clueless. Maybe that\'s why he has Dr. Phil on, to explain some facts of life to him regularly. Larry will nod, but he doesn\'t understand any more than when Billy Graham used to talk to him. King also is becoming famous for the \" softball \" interview, meaning he asks no hard questions. That is a lot due to the fact that most his guests are of his political persuasion. People know being on King\'s show is liking having an hour public relations gig. What all this has meant is a serious decline in ratings the past five years. People see through him and his LiberalandtabloidTV mentality and switched over from King and CNN to Fox News."

text = "Mr. Bean has shaped the face of British TV comedy. He has proved that you do not need wicked words or wit, a massive budget, a great deal of intelligence or even any intelligence to make something brilliant. And Mr. Bean is one of those characters who you just can't forget."

text = preprocessText(text)

mg = MutantGeneration(text)
print(mg.getTemplate())
print(len(mg.getMutants()))





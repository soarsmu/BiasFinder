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
from utils import getMaleNamesAndTheirCountries, getFemaleNamesAndTheirCountries

class CountryMutantGeneration:
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

        # self.coreferences = []
        
        # for r in self.docs._.coref_clusters :
        #     coref = Coreference(r.main, r.mentions)
        #     if self.isValid(coref) : # only take valid coreference
        #         template = self.generateTemplate(coref)
        #         self.templates, self.mutants, self.names, self.countries = self.generateMutant(coref, template)
        #         break

                # self.coreferences = []
        self.person_coreferences = []
        self.person_coreferences = self.getPersonCoreferences()
        if len(self.person_coreferences) == 1 :
            coref = self.person_coreferences[0]
            if self.isValid(coref) :
#                 print("XXXXXX")
                template = self.generateTemplate(coref)
                self.templates, self.mutants, self.names, self.countries = self.generateMutant(coref, template)


    def getPersonCoreferences(self) :
    
        coreferences = []
        for r in self.docs._.coref_clusters :
            coref = Coreference(r.main, r.mentions)
            if self.isPersonCoref(coref) : # only take valid coreference
                coreferences.append(coref)
        
        return coreferences

    def isPersonCoref(self, coref):
        placeholders = []
#         print("COREF: " + str(coref.getReferences()))
        for phrase in coref.getReferences() :
            if phrase.isGenderPronoun():
                return True
            elif self.isPersonName(phrase.getPhrase()):
                return True
            elif phrase.isContainGenderAssociatedWord() :
                return True
        return False

        
    def getOriginal(self):
        return self.original
    
    def getCoreferences(self):
        return self.coreferences
    
    def getTemplates(self):
        return self.templates
    
    def getNames(self):
        return self.names

    def getCountries(self):
        return self.countries
    
    def getMutants(self) :
        return self.mutants
    
    def isValid(self, coref):
        placeholders = []

#         print("COREF: " + str(coref.getReferences()))
        gender = ""
        
        has_gender = False
        has_name = False
        for phrase in coref.getReferences() :
            if phrase.isGenderPronoun():
                if gender == "" :
                    gender = phrase.getGender()
                elif gender != phrase.getGender() : ## there is 2 gender pronoun detected
                    return False
                coref.setGender(phrase.getGender())
                has_gender = True
                placeholders.append(phrase.getPhrase())
            elif self.isPersonName(phrase.getPhrase()):
                placeholders.append(tag(NAME))
                has_name = True
            else :
                placeholders.append(phrase.getPhrase())
                
        if not (has_gender and has_name) :
            return False
        
        coref.setPlaceholders(placeholders)

        ## replace <name><name> into <name>
        ## how if all pronoun
        return True

        
    def isPersonName(self, text) :
        return text in self.person_entities 

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
    
    def getSubstitublePlaceholders(self, placeholders):
        ps = []
        for p in placeholders :
            if "<" == p[0] and ">" == p[-1] :
                ps.append(p)
        return ps
    
    def generateMutant(self, coref, template) :
        templates = []
        mutants = []
        names = []
        countries = []
        
        self.examples = []
        placeholders = coref.getPlaceholders()
        
        
        used_placeholders = set(self.getSubstitublePlaceholders(placeholders))
        
#         print("USED PLACEHOLDERS: ", used_placeholders)
        
        if len(used_placeholders) == 1 :
#             print("XXXXX")
            placeholder = list(used_placeholders)[0]

            gender = coref.getGender()

#             print("PLACEHOLDER: ", placeholder)

            if gender == "male" :
                mutants, names, countries = self.generateMaleMutant(template, placeholder)        
            elif gender == "female" :
                mutants, names, countries = self.generateFemaleMutant(template, placeholder)        
        
            
            if len(mutants) > 0 :
                templates = [self.template] * len(mutants)    
                self.examples.append(mutants[0])
                
        return templates, mutants, names, countries

    
    
    def generateMaleMutant(self, template, placeholder) :
        
        mutants = []

        names, countries = getMaleNamesAndTheirCountries()
        
        for name, country in zip(names, countries) :
            mutant = template.replace(tag(NAME), name.title())
            mutants.append(mutant)

        return mutants, names, countries
                
    def generateFemaleMutant(self, template, placeholder) :
        
        mutants = []

        names, countries = getFemaleNamesAndTheirCountries()
        
        for name, country in zip(names, countries) :
            mutant = template.replace(tag(NAME), name.title())
            mutants.append(mutant)

        return mutants, names, countries
    
    
# there is a person name
# text = "When Nathaniel Kahn embarked into this voyage, he hardly knew who his father really was. By the end of the film, he found him and comes to terms with the strange life he lived as a child."

# It contain a person name and has a salutation
# text = "Meek and mild Edward G. Robinson (as Wilbert Winkle) decides to quit his bank job and do what he wants, open a ”fix-it” repair shop behind his house. Mr. Robinson is married, but childless; he has befriended local orphanage resident Ted Donaldson (as Barry)"

# text = "I'm sorry, but \" Star Wars Episode 1 \" did not do any justice to Natalie Portman's talent ( and undeniable cuteness). She was entirely underused as Queen Amidala, and when she was used, her makeup was frighteningly terrible. For \" Anywhere But Here, \" she sheds her godawful makeup and she acts normally. And not only can she act good, she looks good doing it. I'm a bit older than she ( she's only 18), and I have little or no chance of meeting her, but hey, a guy is allowed to dream, right? Even though Susan Sarandon does take a good turn in this movie, the film belongs entirely to Portman. I've been a watcher of Portman's since \" Beautiful Girls \" ( where she was younger, but just as cute). There's big things for her in the future. I can see it."

# text = "In this film I prefer Deacon Frost. He's so sexy! I love his glacial eyes! I like Stephen Dorff and the vampires, so I went to see it. I hope to see a gothic film with him. \" Blade \" it was very \" about the future \". If vampires had been real, I would be turned by Frost!"

# text = "Mr. Bean has shaped the face of British TV comedy. He has proved that you do not need wicked words or wit, a massive budget, a great deal of intelligence or even any intelligence to make something brilliant. And Mr. Bean is one of those characters who you just can't forget."

# the gender associated word
# text = "Even the manic loony who hangs out with the bad guys in ”Mad Max” is there. That guy from ”Blade Runner” also cops a good billing, although he only turns up at the beginning and the end of the movie."

# the main reference is a pronoun
# text = "This movie is about a man who likes to blow himself up on gas containers. He also loves his mommy. So, to keep the money coming in, he takes his act to Broadway. See! Cody Powers Jarrett blow himself up on his biggest gas container yet!"

# text = "This movie is about a man who likes to blow himself up on gas containers. He also loves his mommy. So, to keep the money coming in, he takes his act to Broadway. SEE! CODY POWERS JARRETT BLOW HIMSELF UP ON HIS BIGGEST GAS CONTAINER YET! TONIGHT! 7.30PM!  However, one day, his mommy dies and Jarrett goes berserk. He kidnaps the audience in the theatre and makes them all stand on top of a huge gas cylinder. Losing control further, he makes them all scream \"MADE IT MA, TOP OF THE CYLINDER!\" in unison. The noise is so deafening that it bursts Jarrets eardrums, causing him to topple from the cylinder into a vat of acid. This Warner Bros. movie is not all it's cracked up to be."

# library problem
# text = "Sometimes Hallmark can get it right - like The 10th Kingdom - but many of their fantasy films plod, and this falls into the latter category. The version I saw may have been cut (a demon [?] shown in the trailer and publicity stills didn't appear), but anything that made the movie shorter can only be a blessing. POSSIBLE SPOILERS IF YOU ARE UNFAMILIAR WITH THE ORIGINAL FAIRY TALE: Anyway, the film updates the story to the early part of the 20th Century (?), and makes Gerda and Kay (here called Kai - being a Lexx fan, I kept expecting him to say, `The Dead do not solve puzzles') 18 year olds. Hans Christian Andersen's basic story is followed: the boy gets a shard of ice in his eye, goes bad, is taken off by the Snow Queen to solve a puzzle in her palace and Gerda goes to find him, having various adventures on the way. As the two main characters are older than in the original, a lot of time is spent getting them together and `in love'. Unfortunately, I was never convinced that they were particularly in love, and certainly not enough in love to make sense of Gerda's quest. By the time the main plot kicks in, the movie's pace has slowed to a crawl. Alas, when Gerda begins her search for Kai, it only manages to pick up the pace to a leisurely stroll. There are a few odd additions to the story that seem to go nowhere. At the start of the film the Snow Queen kills Gerda's mother, but no explanation for this is given. A polar bear living in the Snow Queen's palace is more than he seems (though this is possibly because the producers realised that the bear's feelings towards the Snow Queen would be OK in a Fairy Tale, but not in a modern film). Again, this is never explained. Also, hints that the Snow Queen has an erotic desire for Kai are dropped, but never followed through. The script is also full of anachronisms that really jar you out of the `fairy tale' mood. The production looks good, though there is evidence of penny-pinching: the Snow Queen's palace is the hotel where Gerda and Kai lived covered in ice. The three main characters are played with varying degrees of success: Kai comes across as bland as does Gerda initially, but once she sets off to find Kai you warm to her. Bridget Fonda looks great as the Snow Queen, but seems to be in a different movie to everyone else. Ultimately, the film is unsatisfying. It looks good, but drags and lacks magic."

# text = "My family watches this movie over and over. Even our 3 year old loves it. I like the \" goodness \" in the movie. Giving the stranger a chance ... showing goodwill to one obviously in need of some unconditional acceptance. The movie gives a feeling of goodwill and victory. One other aspect of the movie that makes it so appealing is the personalities of Velvet's siblings. The bird lover. The bug lover. The boy lover! Very cute and happy movie. There is one thing, however that is irritating about it. That is how Mrs. Brown often makes Mr. Brown look foolish or unwise. She, at times, comes off as a know it all, and he as a dimwit, which he is not. Too bad to put that into such a nice story. Nevertheless, we will continue to enjoy this wonderful, old movie!"

# text = "i guess if they are not brother this film will became very common. how long were they can keep this? if we were part, what should they do?so natural feelings, so plain and barren words. But I almost cried last night blood relationship brotherhood love knot film.in another word, the elder brother is very cute.if they are not brothers, they won't have so many forbidden factorsfrom the familysocietyfriendseven hearts of their own at the very beginning. The elder brother is doubtful of whether he is coming out or not at the beginning .maybe the little brother being so long time with his brother and even can't got any praise from his fatherthis made him very upset and even sad, maybe this is a key blasting fuse let him feel there were no one in the world loving him except his beloved brother. and i want to say, this is a so humannatural feeling, there is nothing to be shamed, you may fell in love your motherbrothersister. Just a frail heart looking for backbone to rely on"

# text = "A fascinating slice of life documentary about a husband and wife and their marriage told through the eyes of their son. We all like to think that our parents lived happy lives, that their marriages were full of fulfilment, love, and happy memories. Sadly many of us know this not to be the case of their own families and that of their parents. This wonderful little documentary is told through the camera lens and emotional perspective of the son of a family that has just experienced the death of their mother. The son being a documentary film maker has filmed his elder family for many years, for as he states \" posterity \". Three months after the death of his mother his father remarries his long time secretary. The suddenness of this occurrence stuns the family and pushes the son to dig into the past lives of his mother and father. What he reveals is a fascinating look into the lives of two rather ordinary people who like so many of their generation married early for the wrong reasons and found themselves stuck in a family life where they found they just had to \" make do \". A wife who found herself at times bitterly lonely and unloved and a husband who buries himself in his work. She and intellectual at heart, he a much simpler individual who seems to find most of his pleasures in the quiet solitude of work. They are obviously wrong for each other, this much is clear. Yet they stick it out, for what? Well that\'s part of the mystery, they clearly show affection for each other at times if not ever much love. You won\'t find any truly shocking disclosures here, aside from infidelity on both sides, which in good part is what makes this such a gem. You really feel that these could be your own parents if circumstances were different and indeed makes one question the lives of ones own parents."

# text = "Larry is a perfect example of the Democratic Party in the United States, of which he is a staunch member. King used to be somewhat fair and unbiased and had a variety of guests on. The Party used to be centrist, too, but that was another era. Now, like, Larry, it\'s Far Left.   At least 90 percent of all the guests on King\'s show in the past year or two are Liberals who sit there and bash President Bush and every Conservative they can think of ..... night after night. Bill Mahar, one of the more viscous ones, isand you can look this upthe most frequent guest in the history of King\'s TV show. You can count on other outspoken Left Wingers to be on King\'s show each week, but don\'t hold your breath waiting for a Conservative. They are few and far between. King was also one of the innovators of the media overkill. That all began with the O.J. Simpson trial. Night after night after night that\'s all you ever saw back in the mid \' 90s. Whatever latest gossip on Anna Nicole Smith, or the Petersen murder case, or Paris Hilton, Britney Spears or some other tabloid subject, you can bet Larry will beat it to death. Sadly, all the other networks do the same thing now. Larry was a leader in that regard. King also has the nerve to sometimes give advice, such as on marriage. I am not kidding; I \' ve heard him say it. The joke is that he has been married and divorced a half dozen times! This man has few scruples, believe me. When it comes to morality, he is clueless. Maybe that\'s why he has Dr. Phil on, to explain some facts of life to him regularly. Larry will nod, but he doesn\'t understand any more than when Billy Graham used to talk to him. King also is becoming famous for the \" softball \" interview, meaning he asks no hard questions. That is a lot due to the fact that most his guests are of his political persuasion. People know being on King\'s show is liking having an hour public relations gig. What all this has meant is a serious decline in ratings the past five years. People see through him and his LiberalandtabloidTV mentality and switched over from King and CNN to Fox News."

# text = "Mr. Bean has shaped the face of British TV comedy. He has proved that you do not need wicked words or wit, a massive budget, a great deal of intelligence or even any intelligence to make something brilliant. And Mr. Bean is one of those characters who you just can't forget."

# text = "When my Mum went down to the video store to rent a film for the night my sister and I learned a lesson, to always company my Mum to the video store! In fact the only reason why she chose it was because Colin Firth was in it and she* cough* thinks he's a good actor! It starts off with some beautiful veiws of Africa and then goes DOWN AND DOWN AND DOWN, AND DOWN. After this film I was very surprised that Colin Firth got as far as he did since this pointless film could destroy any actors career. The story is about a divorced women who's son is trying to matchmake her to a man called Matthew Fields who he met whilst impressing his friends because of his large house. Nimi the divorce does not like Matthew at all and is going out with the local vicar who does not like her son John .... and the same with him! I am sorry if you disagree with me and i hope i haven't offended you but to all the people who haven't seen this film, I leave you with one word of warning, DON'T WATCH THIS FILM!!!!!"

# text = "Alright, we start in the office of a shrink, and apparently not a very good one. The main hero from the first Jack Frost is in the shrinks office blurting out random rhymes about Jack Frost. Gee, alright my brother is yelling ' ' Turn it off! ' '. Anyway, back to the crappy movie. The shrink has his speaker phone on and is letting his secretary and her friends listen in on this heroic insane sheriff. I suppose he is supposed to be the hero from the first movie, but he looks nothing like him!. Yadda yadda yadda, they laugh at the poor sheriff, yadda yadda. Now some people are digging up the antifrozed snowman, yadda yadda, now we're in a lab with some type of doctor people .. I don't quite see how this has to do anything, but their poking the antifreeze/ Evil killer mutant snowman with needles, heating it, shocking it, adding strange and bizarre chemicals to it, the whole nine yards. Nothing. Alright, they give up and leave it in a fish tank. One of the doctors leaves his coffee on the top of the tank. The janitor walks in, cleans stuff, bumps the fishtank and the coffee spills the tank which makes Jack alive. Behold the power of mocha! Now somehow he is in .. uh .. i believe the Bahamas ... but it looked more like Hawaii .. But it couldn't be Hawaii! Unless they spent all of their budget on the dang air plane tickets. Bah .. I wo nt spoil the rest of this rotten movie, so you'll have to rent it and watch it your self ... Er ... i wouldn't suggest doing so though .... Sheesh .."

# text = "A young woman, Jodie Foster, is witnessing a mafia murder, reports the killing to the local police, and becomes herself a hit target by the mob operatives. A professional killer, Dennis Hopper, hired by mafia, is stalking her to prepare for the hit, but eventually he falls for her. Then, as a parody of the Stockholm Syndrome that defines a case when an abducted hostage begins to like and cooperate with the kidnapper, Jodie Foster falls for her abductor too, make love, and both prepare for a getaway. Denis Hopper, the actor, tries to align himself with the creative ambitions of Dennis Hopper, the director. The result is disappointing, and fails to keep pace with the artistic level of a great performer as Dennis Hopper is. There is no real thrill and the script is sometimes naive and predictable. The film is saved to some extent by the performance of Jodie Foster who is not at her best, but still shines with her talent, beauty and gift. Of historical interest is the short appearance of Vincent Price, and, in a small act, of Charlie Sawn known from his great part in \" Wall Street \". If you decide to spend the 116 minutes to see the film, it is not a complete loss; this movie offers easy entertainment, but we would expect much more from the director of \" Easy Rider \", and the actress who gave us the character of Sarah Tobias in \" The Accused \."


# text = "** MAJOR SPOILERS** Watchable only for the action sequences not the story or acting in it \" Nature Unleashed: Fire \" has one of the longest and excruciating endings in modern motion picture history. We have the fearless Ranger Jake, Bryan Genesse, leading this trio of hysterical bikers to safety in of all paces an explosive fume beaching mine shaft! This during a raging forest fire! It seems that Ranger Jake with all his knowledge of the great outdoors didn't realize that a mine shaft that's leaking with dangerous and explosive methane gas is the last place to go when all the woods around it is on fire!*** SPOILERS FROM THIS POINT ON*** All this started some time ago when Ranger Jake in an effort to save the not that on the ball miner Tiny, Chris Harz aka \" The Sherd \", let him slip through his fingers and fall to his death at the bottom of the mine shaft, or did he! Even though we were kept in suspense to who's setting the forest fires for the first half of the movie it wasn't a surprise at all the Tiny was the culprit! As you would expect in movies like these Tiny seemed to be made of hardened steel in that nothing that ever happened to him, fires explosions as well as impaling, could stop the crazed miner. Before Tiny's reappearance, or resurrection, Ranger Jake got involved in rescuing bikers Chris Mel Sharon & Marcus, Josh Cohen Melanie Lewis Anastasia Griffith & Ross McCall, who were trapped in the woods with fires breaking all around them. Having the usual know it allMarcusamong the bikers things don't go as smoothly as Ranger Jake wanted them to go. Marcus not only eggs on the meek Chris to do something stupid, jump with his bike over a 10 foot pile of logs, but has the guy break his leg. This makes it almost impossible for Ranger Jake to have Chris airlifted out before the fires consume him as well as his fellow bikers! For the remainder of the movie Ranger Jake, who put himself in charge, makes boner after boner in his attempt to save himself and the trapped and lost in the woods bikers! All this ends with Jake's brilliant idea to hide in a dangerous and abandoned mine shaft with the rescue party just yards away from rescuing them if they only stayed put and in the open where the rescue team could find them! Even though he was supposed to be the life of the party, or movie, Tiny for all his efforts in being another indestructible super villain came across as a man who spent too much time out in the sun. The makeup job on Tiny was so outrageous that he looked like he dumped a jar of spaghetti sauce over his head instead of having it burned to a crisp. Ranger Jake came across as either somewhat very naive or retarded in his being so taken in by the dangerous Tiny in always trying to save the rampaging psycho who never hid his feelings about what he had in mind for the play by the rules Forest Ranger. In fact Ranger Jake actually encouraged Tiny to do both him and the bikers in by showing him how incompetent he was in trying to save them. The fact that Ranger Jake was successful wasn't because he was so smart but because Tiny, despite his indestructibility, was so brainless!"
text = "A young woman, Jodie Foster, is witnessing a mafia murder, reports the killing to the local police, and becomes herself a hit target by the mob operatives. A professional killer, Dennis Hopper, hired by mafia, is stalking her to prepare for the hit, but eventually he falls for her. Then, as a parody of the Stockholm Syndrome that defines a case when an abducted hostage begins to like and cooperate with the kidnapper, Jodie Foster falls for her abductor too, make love, and both prepare for a getaway. Denis Hopper, the actor, tries to align himself with the creative ambitions of Dennis Hopper, the director. The result is disappointing, and fails to keep pace with the artistic level of a great performer as Dennis Hopper is. There is no real thrill and the script is sometimes naive and predictable. The film is saved to some extent by the performance of Jodie Foster who is not at her best, but still shines with her talent, beauty and gift. Of historical interest is the short appearance of Vincent Price, and, in a small act, of Charlie Sawn known from his great part in \" Wall Street \". If you decide to spend the 116 minutes to see the film, it is not a complete loss; this movie offers easy entertainment, but we would expect much more from the director of \" Easy Rider \", and the actress who gave us the character of Sarah Tobias in \" The Accused \"."
# text = "Columbo is guest lecturer for a criminology class. The students invite him along for their afterclass gettogether. Transiting the nearby parking garage, they discover their regular teacher, next to his car, dead from a gunshot wound. ( No, Columbo was not after the man's job.) As a class project, Columbo involves the students in his sleuthing. Two students, tentatively identified by the viewer as culprits, were in the lecture hall for the entire class. Furthermore, surveillance camera tapes of the parking garage show that no one other than the professor entered or left after he was last seen unexpectedly departing the lecture hall. Reversing the normal routine, Columbo is the one that is pestered by the evil (?) duo, eager for progress reports and an ear for their theories. Forensic evidence is almost nonexistent. Solution of the case hinges on some eventual and interesting good luck. On first viewing, it seemed that Columbo had swallowed whole the culprits ' misdirection; however, on repeat viewing, small details revealed that not to have been the case at all. This reviewer has yet to tire of \" Columbo Goes to College. \""
# text = "I guess it wasn't entirely the filmmaker's fault though. The film suffered from the unimaginably stupid decision to tell Clayton Moore ( who had done the role in the 1950's and was the Lone Ranger us old folks grew up with) he couldn't wear the mask in public. Now mind you, the poor guy wasn't making all that much money doing so, and it wasn't like he was going to take anything away from this film, but the whole thing seemed ... gratuitous. The other thing the film suffered from ( besides a leading man whose voice was so awful they had to overdub it) was that fact that Westerns weren't so hip in 1981. John Wayne was dead and we had just been subjected to a decadelong major liberal guilt trip about how the west was built on genocide of the Native Americans. ( That and Blazing Saddles sent up the whole genre! The Campfire scene. Enough said!) Hollywood shied away from Westerns, because Science Fiction was COOL then. The one scene that underscored it was when after rescuing the drunken President Grant ( and seriously, I'd have let Grant stay with the bad guys. The country would have been better off!) Grant asks Tonto what his reward should be \" Honor your treaties with my people \". Yeah, right, like THAT was going to happen!"
# text = "The Paul Kersey of DEATH WISH 3 is very far removed from the Paul Kersey of the original film. If you remember the 1974 film then you will remember Kersey was a \" Conchie \" during the Korean war and that he was physically sick after he committed his first execution. Ten years later Kersey seems to have learned unarmed combat and how to handle anti tank weapons in his spare time. But I`ll overlook that gaffe because DW3 is the best of the sequels, lowlife scum bags get shot dead, burned alive, their teeth smashed, and thrown to their deaths by middle aged housewives armed with sweeping brushes. Yeah I know the gang members are multi ethnic and for that they deserve some credit but even if they`re not racist they`re still murdering scum who deserve all they get from Kersey and the innocent citizens. Who needs Mayor Rudy when you`ve got Paul Kersey, an anti tank rocket and a bunch of old age pensioners to reclaim the streets from the criminal creeps. Paul Kersey I salute you sir"
# text = "Chuck Jones's ' Odorable Kitty ' is the cartoon that introduced Albercik to the world sort of. There are a few key differences between the Pepe we know and love ( or hate, in the case of some people) and the character in this cartoon. For one, the disguised cat who Albercik amorously pursues in ' Odorable Kitty ' is distinctly male. Also, Albercik is exposed as a fraud whose real name is Henry at the cartoon's climax, his French accent dropping away when his wife and family turn up. Albercik is not even the lead character here, the focus favouring the putupon cat who disguises himself as a skunk to scare off his enemies. For the most part, the storyline largely follows the usual format of a Albercik cartoon but Albercik 's aggressive courtship is lacking the usual wisecracks and straight to camera addresses that make him such a great character. He is also not nearly as handsome as he would become and rather awkwardly animated. In fact, ' Odorable Kitty ' is a fairly ugly and clumsy looking cartoon all round. Its main source of appeal comes from its concept which was original at this stage before it became the template for every Albercik cartoon that followed. This subsequent development has robbed ' Odorable Kitty ' of any impact whatsoever and to modern viewers it just looks like a rather dull Albercik short with a weird surprise ending. As a child, I hated Albercik . As an adult, able to appreciate his more sophisticated, verbal and risqu humour, I love the character and most of his cartoons. ' Odorable Kitty ' makes me feel like a child again!"
# text = "*** SPOILERS****** SPOILERS*** There's not much that can be said about this earlytalkie era flick. ( I'm hesitant to call \" Cimarron \" a \" film \", because I feel that the word is too esoteric.) But what can be said about it ... mainly speaks against it. Take, for example, the overuse of portraying Indians as bad folk. In one scene, the little boy of the flick's lead character -- an overbearing and overambitious family man who wants to set up a newspaper business-- is playing just outside of his father's office. An Indian kneels down in front of the child. \" Hello, \" the boy says to the Indian, in a very polite manner. The Indian gives him a feather, stands up and walks off. Yancy Cravat, Jr. excitedly runs inside the office. \" Mommy! Mommy! \" he shouts, holding up the Indian's gift. \" Look what an Indian gave me! \" \" How many times do I have to tell you! \" she snaps at the boy. \" You aren't ever to talk to those filthy Indians! \" Yancy Yates, Sr. ( Richard Dix) comes across as a man who speaks with a forked tongue. At the start of the story, he seems to have a definite plan for giving his family a better life. But, we soon enough discover, he's no great over achiever -- much less a totally goodmoral minded man. His slave child, Isaiah ( Eugene Jackson) is one telltale sign of this. Upon his family's first trek to a Sunday morning church service -- one at which, curiously enough, Cravat is to give the sermon -- Isaiah tries to come along, dressed up like Cravat, longtail suit, holster, gun and all. Cravat tells him to go home. \" Ya ' all doesn't want me to come with ya ' ta church? \" Isaiah says with a pout. \" No! \" Cravat corrects him, patting him on the shoulder. \" You don't understand! I want you to stay and guard the house. And if anyone at all comes along ... you shoot him dead! \" The characters -- not to mention the actors -- in \" Cimarron \" couldn't act their way out of burlap sacks, despite their obvious efforts. And nothing in the script was any too commendable, either. ( Granted: the incomparable Edna May Oliver -- notorious for playing the Red Queen in Alice In Wonderland, also released in 1931-actually manages to look good, pulling off her portrayal of a pompous old woman, which is what she's also been bestknown for.) But, aside from that, well ... Yancy Yates isn't popular in town from the first week he arrives, and one of the outlaws decides to shoot Cravat's white hat off as he and his wife ( Irene Dunne) are casually walking by. Despite her anger with the man who fired the bullet, Cravat just takes it completely in stride. Not only was this story not \" shooting for realism \", but it was very lacking in several key areas: e.g., Cravat's newspaper isn't ever really seen. ( Bulletins and posters, yes -- but not any newspaper.) Perhaps strangest of all, though: this is set in a small town in Kansas. Yet, for some reason or other, Yancy Cravat is deadset on calling his paper \" The Oklahoma WigWam. \" Really good westerns have always been very few and far between -- the only exceptions being Clint Eastwood's socalled \" spaghetti westerns \" of the late ' 60s to early ' 70s. Cliche westerns, on the other hand, are a dimeadozen. If you like cliche westerns, \" Cimarron \" will do you proud -- but, as for me ... it did me embarrassment."
# text = "Richard Schickel's 1991 documentary about Gary Cooper\" Gary Cooper: American Life, American Legend \" gives us a look at the tremendous, allAmerican star through his films and his life. Narrated by Clint Eastwood, the theme is definitely \" Gary Coooper, American \" as we are taken through fast clips of his many appearances in westerns, and scenes from \" Meet John Doe, \" \" Mr. Deeds Goes to Town, and \" Sgt. York. \" The best part of the documentary is the home movies of Cooper and his family as well as his childhood photos, showing him as a beautiful blonde kid with the sunny smile he would have his entire life. There is also a hilarious clip of Cooper on \" The Jack Benny Show \" doing the comeback on the number \" Bird Dog \"and Benny loses it. The documentary also takes us briefly through his tumultuous affair with Patricia Neal, which nearly ruined both their lives. There's a certain cohesiveness missing from this bio/ retrospectiveit jumps around a lot and has no footage of Cooper being interviewed, which would have added a lot. Also, Clint Eastwood's narration was described as unobtrusive. What it was, was boring and monotone. Given that Cooper himself tended to be the strong, silent type on screen, we could have used a little animation. On a personal note, Gary Cooper was one of the handsomest men who ever livedthere were some looks at him in his early films, but not nearly enough for this fan. That smile, those lips, that bone structurehe was handsome throughout his life, but in films like \" Morocco \" and \" Desire, \" he is devastating. Instead of sitting through a scene from one of his worst performances, as Howard Roark in \" The Fountainhead, \" giving a speech that he admitted to the author he did not understanda young, suave Cooper in a tux would have been a nice touch. This documentary, alas, was definitely produced by a man."
# text =  "Man with the Screaming Brain is a story of greed, betrayal and revenge in the a small Bulgarian town. William Cole, wealthy industrialist, winds up with part of his brain replaced by that of a Russian cab driver Yegor. The two couldn't be more different, but they share one thingboth were killed by the same woman. Brought back to life by a mad scientist, William and Yegor form an unlikely partnership to track down their common nemesis. Bruce Campbell returns to the B horror movie genre that gave him his cult status, this time not only in front of the screen, but behind the lens. Unfortunately for this time around, the laughs don't deliver and Campbell has to resort to what he does best to try and fill the gap in this film. As a fan of Campbell, who has the movies, the books and the action figures, I was hoping for another hit to add to my collection. Although, after seeing this film before the purchase, I am glad that I don't have the \" pleasure \" of adding it. The film first goes wrong in the story, which at first sight, seems like harmless fun but turns out to be boring drawn out dribble. Which is a sad thing to say because it was written by Mr. Campbell himself. The comedy never really hits, it only makes us scratch our heads. It seems that Campbell ran out of things that are funny and resulted in giving the audience what we've already seen ... him fighting himself. Ted Raimi, the brother of Evil Dead director Sam Raimi, is undoubtedly the highlight of the film. He brings a freshness to it and an entertaining time when the film really needs it. It helps if you are a fan and have been following these stooges from Evil Dead to Xena, which is why I felt compelled to like this film. Campbell's experience as a director, from directing episodes of the TV series Hercules is apparent. Campbell makes the film work well enough, even with the lowbudget. In the end, there aren't as many things going for this as one would hope for, but the fans of Campbell will stick behind it no matter what, unfortunately for this fan ... I won't."
# text = "I have to admit that when first saw Madonna performing Holiday on Top of the Pops many years ago I said to my wife \" another American one hit wonder getting the whole thing wrong!! \" Well she was wearing a fright wig and was appallingly dressed. I have never grown to love her the way my daughter does but I have to eat my words. I do like some of her stuff and sometimes enjoy her filmed concerts. This Confessions tour film is great, even if the music is not(and its not). I was impressed by the staging and concepts. Madonna's own performance was enhanced by the incredible dancers she chose to support her. My daughter was at the London gigs and was crazy about it. The lady ( Madge) has proved my initial assessment of her so very wrong!!"
# text = "Every movie Quentin Tarantino has made has become progressively worse. I'd like to believe that most people would agree with that statement, but seeing as \" Inglourious(sic) Basterds(sic) \" has an 8.5/10 from over 100,000 ratings, it doesn't seem like the general moviegoing public has any sense. Even his best work, Reservoir Dogs, wasn't a ' masterpiece. ' The trouble is that claiming that you like Tarantino's work has become trendy. As soon as that happens, you get boatloads of people ready and willing to hop on another bandwagon. They will ignore laughably terrible acting, and utterly selfindulgent writing just so they can be part of the exclusive club called \" everyone. \" This movie is so terrible, that I swear it must be some sort of twisted joke by Tarantino to see how much torture his fans will tolerate and still praise him. Like another reviewer has already said: \" Previous Tarantino movies were from a guy in love with other movies. This one is from a guy in love with his own writing. \" I couldn't agree more. This movie is nothing more than selfindulgent and injoke riddled writing paired with acting ability taken right out of a high school play. But, thanks to the general movie going public, I'm sure it will still go down as one of the best movies ever made. Bravo, Tarantino. You've pulledoff one of the best practical jokes of all time."
# text = "Meester Sharky, you look so ... normal. You would never get a table in this fancy cocktail restaurant/ bistro. I, on the other ' and eat grapes and pate ' ere every day. You like my fur coat with all the fine trimming? My enormous golden rings of gold? Or maybe you like these blonde, ' ow you say?, bombshells, who are all qualified in aerobics and naked petanques, who decorate my long, maroon velvety sofa like so many soft boiled larks on a plate of pan fried foie gras and figs. You like? You can't have! Zey are all mine. You will never possess ' er as I possessed ' er. Domino was the best, apart from Maman. You do not understand the art of lovemaking. Just look at your inferior moustache. It is almost funny to me, non, to think of that ludicrous protuberance on your silly face, as you snuffle around Domino's love hillock like the piggy seeking the truffle in the forest, the forest heaving and swaying in the hot winds of desire! You lose again Sharky. When I make love to the women zey know, Sharky, zey know. Zey learn, zey learn until zey become the teacher. Not nanomaths, the arts of love. Domino was the seedling which I watered. I watered her so very often. Everywhere Sharky. Her scented petals, her proud stalk, everywhere. She will wither under your ridiculous hose, like the souffl removed from the oven five minute too soon. I must go now Sharky, you bore me so with your disgraceful behaviour. It is you who will be flushed down le pissoir like the smelly thing. Bon chance!"
# text = "I saw Bogard when it was released in the 70s. It was one of those pictures that received an X rating for violence. We snuck into our local grindhouse, and saw it anyway. Pretty good picture. Lots of blood from the street fights, although the cheap sound effects for the punches took something away from it. And lots of sex. I remember one of the early scenes when Bogard meets this pretty brunette in an apartment she is showing him. Without saying a word, he picks her up, puts her in the windowsill and nails her. From what I remember the picture sailed from that point on. So, when I found out Bogart is also called Black Fist and was available on VHS, I ordered it online. I was very disappointed. Black Fist is Bogard edited for television. So many of the scenes I remember were missing, I wondered if indeed, this was the same picture."
text = "This is a wonderful family sitcom. Rowan Atkinson has appeared in to other excellent sitcoms, The Thin Blue Line ( Better than this) and Blackadder ( Not better than this). Mr Bean is a no talking, human disaster. He goes to places and gets himself in absolute mayhem, the mayhem includes: Climbing up to the top diving board and is too frightened to jump off, taking about 20 minutes, until some kids eventually throw him off, ending up inside a washing machine and driving his car while sat on a roof. Bean drives a Mini and has a teddy. This was quite similar to The Baldy Man, a series staring Gregor Fisher who says very little, but gets himself in mayhem Best Episode: Do it Yourself Mr Bean, Episode 9: Bean hosts a New Years Eve party, then gets some stuff for decorating his flat, but has too much stuff and has to drive his car on the roof."
# text = "Ever sense i was a kid i have  loved this movie. i have always been a fan of Joseph Mazzello. the kid had pure talent in both this movie and Jurassic Park. I have been looking for the DVD or VHS to purchase at a store near me i ca nt seem to find it i hope it goes on DVD! well anyways great movie. If anyone knows where i can find this please contact me at wrp24@adelphia.net. Also can anyone really explain what happened with bobby. was her real or was he fake and was he mikes imaginary friend and his escape? lol I'm clueless. my favorite part had to be definitely where they made the monster juice and spilled it all over the kitchen its funny but also a sad part as well because of what happens to bobby due to the mess .. i would've liked to see the boyfriends face because he played his part pretty good. i think the mother was a great actress i think her name is Lorraine Bracco or some sorta name like that .. well that s all please contact me wrp24"
# text = "Luc Besson's first work is also his first foray in science fiction, a genre to which he will return fourteen years later with \" the Fifth Element \" ( 1997). Even if this film was strongly influenced by Hollywood cinema, it is still highly enjoyable. Back in 1983, \" le Dernier Combat \" reveals Besson's own approach of science fiction. He takes back a threadbare topic and his efforts are discernible to make a stylish work. Shot in widescreen and black and white, a disaster has destroyed virtually all the population from earth and we will never know what was this disaster and why men can't talk any more. Some barbarian hordes were formed. In parallel, a man ( Pierre Jolivet) lives on his own and arrives in an unrecognizable Paris where he is received by a doctor ( Jean Bouise). There are no words in Besson's work. The characters ' actions and the progression of the events go through looks and gestures. Although the starting point and the backdrop are unnerving, the film has never the look of a despondent one. It seems that the man and the doctor try to reproduce gestures and actions linked to mankind before the disaster. The film opens with the man having sex with an inflatable doll. Later, the doctor tries to make him speak through a machine and he is a painter in his spare time. It's all the more intriguing as these paintings seem to come from the prehistoric times. Following this reasoning, one could argue that the bearded giant ( Jean Reno) embodies evil and a threat to the efforts deployed by the man and the doctor to regain what finally made a human being. Ditto for the gang of baddies at the beginning of the film. The pessimistic whiff that such a film could convey isn't really at the fore and gives way to a glimmer of hope. Personally, the film could have gained with no music at all, except the one the man can hear with his cassette recorder. Luc Besson was to make better and still entrancing films like this one, he also boosted Pierre Jolivet's career as a director who will leave a patchy work behind him in the future: \" Force Majeure \" ( 1989), \" Simple Mortel \" ( 1991), \" ma Petite Entreprise \" ( 1999) or \" Filles Uniques \" ( 2003)."


text = "A young woman, Jodie Foster, is witnessing a mafia murder, reports the killing to the local police, and becomes herself a hit target by the mob operatives. A professional killer, Dennis Hopper, hired by mafia, is stalking her to prepare for the hit, but eventually he falls for her. Then, as a parody of the Stockholm Syndrome that defines a case when an abducted hostage begins to like and cooperate with the kidnapper, Jodie Foster falls for her abductor too, make love, and both prepare for a getaway. Denis Hopper, the actor, tries to align himself with the creative ambitions of Dennis Hopper, the director. The result is disappointing, and fails to keep pace with the artistic level of a great performer as Dennis Hopper is. There is no real thrill and the script is sometimes naive and predictable. The film is saved to some extent by the performance of Jodie Foster who is not at her best, but still shines with her talent, beauty and gift. Of historical interest is the short appearance of Vincent Price, and, in a small act, of Charlie Sawn known from his great part in \" Wall Street \". If you decide to spend the 116 minutes to see the film, it is not a complete loss; this movie offers easy entertainment, but we would expect much more from the director of \" Easy Rider \", and the actress who gave us the character of Sarah Tobias in \" The Accused \."

# text = preprocessText(text)

# mg = CountryMutantGeneration(text)
# print(mg.getMutants())
# print(mg.getTemplate())
# print(len(mg.getMutants()))





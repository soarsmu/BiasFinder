import pandas as pd
from names import mnames, fnames

#mutant, template, person, gender, emotion, label
def getLabel(emotion):
    if emotion == "joy":
        return 1
    else:
        return 0


templates_df = pd.read_csv("./templates.txt", sep = "\n", names = ["template"])
male_names = mnames
female_names = fnames
emotional_state_words = {"anger": ["angry", "annoyed", "enraged", "furious", "irritated"], "fear": ["anxious", "discouraged", "fearful", "scared", "terrified"],"joy": ["ecstatic", "excited", "glad", "happy", "relieved"], "sadness": ["depressed", "devastated", "disappointed", "miserable", "sad"]}
emotional_situation_words = {"anger": ["annoying", "displeasing", "irritating", "outrageous", "vexing"], "fear": ["dreadful", "horrible", "shocking","terrifying", "threatening"], "joy": ["amazing", "funny", "great", "hilarious", "wonderful"], "sadness": ["depressing", "gloomy", "grim", "heartbreaking", "serious"]}
df = pd.DataFrame(columns = ["label", "mutant", "original_template", "template", "person", "gender", "emotion"])
vowels = ('a','e','i','o','u')

for index,row in templates_df.iterrows():
    template = row["template"]

    for name in male_names:
        template_with_name = template.replace('<person>', name.capitalize())
        if '<emotional state word>' in template_with_name:
            for key, array in emotional_state_words.items():
                for word in array:
                    mutant = template_with_name.replace('<emotional state word>', word)
                    template_with_emotion = template.replace(
                        '<emotional state word>', word)
                    label = getLabel(key)  
                    df.loc[len(df)] = [label, mutant, template,
                                       template_with_emotion, name.capitalize(), "male", key]
        elif '<emotional situation word>' in template_with_name:
            for key, array in emotional_situation_words.items():
                for word in array:
                    mutant = template_with_name.replace('<emotional situation word>', word)
                    template_with_emotion = template.replace(
                        '<emotional situation word>', word)
                    if 'himself/herself' in mutant:
                        mutant = mutant.replace('himself/herself', 'himself')
                        if word.startswith(vowels):
                            mutant = mutant.replace('a/an', 'an')
                        else:
                            mutant = mutant.replace('a/an', 'a')
                    label = getLabel(key)  
                    df.loc[len(df)] = [label, mutant, template,
                                       template_with_emotion, name.capitalize(), "male", key]
                    
    for name in female_names:
        template_with_name = template.replace('<person>', name.capitalize())
        if '<emotional state word>' in template_with_name:
            for key, array in emotional_state_words.items():
                for word in array:
                    mutant = template_with_name.replace('<emotional state word>', word)
                    template_with_emotion = template.replace(
                        '<emotional state word>', word)
                    label = getLabel(key)  
                    df.loc[len(df)] = [label, mutant, template,
                                       template_with_emotion, name.capitalize(), "female", key]
        elif '<emotional situation word>' in template_with_name:
            for key, array in emotional_situation_words.items():
                for word in array:
                    mutant = template_with_name.replace('<emotional situation word>', word)
                    template_with_emotion = template.replace(
                        '<emotional situation word>', word)
                    if 'himself/herself' in mutant:
                        mutant = mutant.replace('himself/herself', 'herself')
                        if word.startswith(vowels):
                            mutant = mutant.replace('a/an', 'an')
                        else:
                            mutant = mutant.replace('a/an', 'a')
                    label = getLabel(key)  
                    df.loc[len(df)] = [label, mutant, template,
                                       template_with_emotion, name.capitalize(), "female", key]

df.to_csv('../../data/eec/gender/twitter_semeval/test.csv',
          header=None, index=None, sep='\t')



import pandas as pd

from utils import generateMultipleMutantOcc

placeholderLoc = "./asset/neutral-occupation.csv"
occupationPlaceholder = pd.read_csv(placeholderLoc)

text = "And one only, in my opinion.<br /><br />That reason is Margaret Leighton. She is a wonderful actress, on-stage as well as on the screen. We have few chances to see her, though. I think that's especially true in the United States.<br /><br />Here she plays a sympathetic role. Not only that but she is also very pretty and meant to be something of a bombshell.<br /><br />Walter Pigeon does not hold up the tradition of Drummond performers. He is always reliable but he's not much fun. He's not a rascal or a knave. \\xa31 Consequently, this seemed to me a talky endeavor with little action or suspense. But check it out for Leighton."

text = "Bela Lugosi is an evil botanist who sends brides poisoned orchids on their wedding day, steals the body in his fake ambulance/ hearse and takes it home for his midget assistant to extract the glandular juices in order to keep Bela\'s wife eternally young. Some second rate actors playing detectives try to solve the terrible, terrible mystery. Bela Lugosi hams it up nicely, but you can tell he needed the money. This film is thoroughly awful, and most of the actors would have been better off sticking to waiting tables, but the plot is wonderfully ridiculous. Tell anyone what happens in it and they tend to laugh quite a lot and demand to see the film. I got the DVD in a discount store 2 for \\xc3 \\x82 \\xc2 \\xa31, which I think is a pretty accurate valuation, anyone paying more for this would be out of their mind. "


x = generateMultipleMutantOcc(text, occupationPlaceholder["occupation"].values.tolist())
# for y in x :
#     print("")
#     print(y)

print(len(x))
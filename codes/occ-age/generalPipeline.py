from utils import generateMultipleMutantAge
from utils import generateMultipleMutantOcc
import pandas as pd
import time


placeholderLoc = "./asset/neutral-occupation.csv"
occPlaceholder = pd.read_csv(placeholderLoc)
occPlaceholderList = occPlaceholder['occupation'].tolist()

datasetLoc = "./asset/test.csv"
# dataset = pd.read_csv(datasetLoc)
dataset = pd.read_csv(datasetLoc, encoding="latin1", sep="\t", names=["sentiment", "review"])
lb = 0 # lower bound
ub = 10 # upper bound
dataset = dataset[lb:ub]
# dataset = dataset.sample(n = 1000, random_state=123)


def generalPipeline(dataset, occPlaceholderList):
    start = time.time()
    finalOutputAge = ()
    finalOutputOcc = ()
    counter = lb + 1
    for index, row in dataset.iterrows():
        print("count: {}".format(counter))
#         print("now processing")
#         print(str(row.review.encode('utf-8')))
#         print()
#         print("processing age")
        outputAge = generateMultipleMutantAge(str(row.review.encode('utf-8')))
        if len(outputAge) != 0:
            for element in outputAge:
                tempElement = ()
                tempElement = element + (row.sentiment,)
                finalOutputAge = finalOutputAge + (tempElement,)
#         else:
# #             print("no age mutant-able sentence")
#         print()
#         print("processing occupation")
        outputOcc = generateMultipleMutantOcc(str(row.review.encode('utf-8')), occPlaceholderList)
        if len(outputOcc) > 0:
            for element in outputOcc:
                tempElement = ()
                tempElement = element + (row.sentiment,)
                finalOutputOcc = finalOutputOcc + (tempElement,)
#         else:
#             print("no occ mutant-able sentence")
        counter += 1
#         print()
    
    print(time.time() - start)
    
    return finalOutputOcc, finalOutputAge



a, b = generalPipeline(dataset, occPlaceholderList)
if len(a) > 0:
    outputDataOcc = pd.DataFrame(list(a))
    outputDataOcc.columns = ['mutant', 'template', 'ori', 'occ', 'placeholder', 'sentiment']
#     outputDataOcc['sentiment'] = outputDataOcc['sentiment'].replace(['negative'],0)
#     outputDataOcc['sentiment'] = outputDataOcc['sentiment'].replace(['positive'],1)
    headers = ['0', '1', 'mutant', 'template', 'original', 'occupation']
    preparedOccDataTemp = [outputDataOcc['sentiment'], outputDataOcc['mutant'], outputDataOcc['mutant'], outputDataOcc['template'], outputDataOcc["ori"], outputDataOcc['placeholder']]
    preparedOccData = pd.concat(preparedOccDataTemp, axis = 1, keys = headers) 
    preparedOccData.to_csv('./prepared/occ.csv', index=False)
if len(b) > 0:
    outputDataAge = pd.DataFrame(list(b))
    outputDataAge.columns = ['mutant', 'template', 'ori', 'name', 'age', 'sentiment']
#     outputDataAge['sentiment'] = outputDataAge['sentiment'].replace(['negative'],0)
#     outputDataAge['sentiment'] = outputDataAge['sentiment'].replace(['positive'],1)
    headers = ['0', '1', 'mutant', 'template', 'original', 'age']
    preparedAgeDataTemp = [outputDataAge['sentiment'], outputDataAge['mutant'], outputDataAge['mutant'], outputDataAge['template'], outputDataAge['ori'], outputDataAge['age']]
    preparedAgeData = pd.concat(preparedAgeDataTemp, axis = 1, keys = headers) 
    preparedAgeData.to_csv('./prepared/age.csv', index=False)


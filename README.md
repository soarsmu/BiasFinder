# BiasFinder: Uncovering Bias in Sentiment Analysis Systems through Metamorphic Testing

### Overview

Artificial Intelligence (AI) systems, such as Sentiment Analysis (SA) systems, typically learn from large amounts of data that may reflect human biases. Consequently, SA systems may exhibit unintended demographic bias based on specific characteristics (e.g., gender, occupation, country-of-origin, etc.)  Such biases manifest in an SA system when it predicts a different sentiment for similar texts that differ only in the characteristic of individuals described. Existing studies on revealing bias in SA systems rely on the production of sentences from a small set of
short, predefined templates. 

To address this limitation, we present **BiasFinder**, an approach to discover biased predictions in SA systems via metamorphic testing. A key feature of BiasFinder is the automatic curation of suitable templates based on the pieces of text from a large corpus, using various Natural Language Processing (NLP) techniques to identify words that describe demographic characteristics. Next, BiasFinder instantiates new text from these templates by filling in placeholders with words associated with a class of a characteristic (e.g., gender-specific words such as female names, “she”, “her”). These texts are used to tease out bias in an SA system. 

**BiasFinder** identifies a bias-uncovering test case when it detects that the SA system exhibits demographic bias for a pair of texts, i.e., it predicts a different sentiment for texts that differ only in words associated with a different class (e.g., male vs. female) of a target characteristic (e.g., gender). Our empirical evaluation showed that BiasFinder can effectively create a large number of realistic and diverse test cases that uncover various biases in an SA system with a high true positive rate.

## Requirements

For fine-tuning SA, we modify some codes from [Xuyige et al](https://arxiv.org/abs/1905.05583) at this Github repository https://github.com/xuyige/BERT4doc-Classification. Xuyige et al. implement the fine-tuning task using pytorch-pretrained-bert package (now well known as transformers). Thus, we need:

+ torch>=0.4.1,<=1.2.0 -> currently torch 1.2.0 with cuda 10.0 is used

For nlp task, please install thess libraries:
+ spacy
+ pandas
+ numpy
+ scikit-learn
+ nltk
+ neuralcoref
+ fastNLP

For occupation bias, you need StanfordCoreNLP and several libraries:
+ inflect
+ pycorenlp -> [Stackoverflow Guide to serve StanfordCoreNLP as an API](https://stackoverflow.com/questions/32879532/stanford-nlp-for-python)


For preparing data from [genderComputer](https://github.com/tue-mdse/genderComputer), please install thess libraries:
+ python-nameparser
+ unidecode

**Tips**: you may use docker for faster implemention on your coding environment. https://hub.docker.com/r/pytorch/pytorch/tags provide several version of PyTorch containers. Please pull the appropiate pytorch container with the tag 1.2 version, using this command.

```
docker pull pytorch/pytorch:1.2-cuda10.0-cudnn7-devel
```

## Setup Dataset and BERT fine-tuning model

This part will tell you the preparation needed. If you don't find any error in this step. You can go to the [end-to-end program](#end-to-end-program). Then read the [Important Resource](#important-resource) and [Technical Report](#technical-report)

### 1) Prepare the dataset 

dataset | description
------------ | -------------
`asset/imdb/` | We use IMDB movie review dataset downloaded from [Google Drive](https://drive.google.com/drive/u/0/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M) proposed by [Zhang et al. (2015)](https://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classification.pdf). 
`asset/gender_associated_word/` | It contains pre-determined values for Gender Associated Words
`asset/gender_computer/` | It contains a notebook `asset/gender_computer/genderComputer/prepare_male_female_names.ipynb` to prepare the names for **BiasFinder** experiment.
`asset/predefined_occupation_list/neutral-occupation.csv/` | It contains pre-determined words for neutral occupations 


### 2) Prepare Google BERT

Please download [BERT-Base, Uncased](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip) model. You will get `uncased_L-12_H-768_A-12` folder. Put it inside `models/`.


### 3) Fine-Tune BERT on SA using IMDB

#### Convert BERT Tensforflow checkpoint into PyTorch checkpoint

Run this command inside the `codes/fine-tuning/` folder.

```
python convert_tf_checkpoint_to_pytorch.py \
  --tf_checkpoint_path ./../../models/uncased_L-12_H-768_A-12/bert_model.ckpt \
  --bert_config_file ./../../models/uncased_L-12_H-768_A-12/bert_config.json \
  --pytorch_dump_path ./../../models/fine-tuning/pytorch_bert_base_model.bin
```

Please make sure that folder `models/fine-tuning/` exists.

#### Fine Tune BERT on SA

Run this command inside the `codes/fine-tuning/` folder.

```
python fine-tune.py \
  --task_name binary \
  --do_lower_case \
  --fine_tune_data_dir ./../../asset/imdb/ \
  --vocab_file ./../../models/uncased_L-12_H-768_A-12/vocab.txt \
  --bert_config_file ./../../models/uncased_L-12_H-768_A-12/bert_config.json  \
  --train_batch_size 24 \
  --learning_rate 2e-5 \
  --num_train_epochs 5 \
  --seed 42 \
  --layers 11 10 \
  --trunc_medium -1 \
  --init_checkpoint ./../../models/fine-tuning/pytorch_bert_base_model.bin \
  --save_model_dir ./../../models/fine-tuning/pytorch_imdb_fine_tuned/
```

**Important Parameter**

Parameter | Description
------------ | -------------
`fine_tune_data_dir` | Data for fine-tuning on downstream task, currently IMDB. You need to put a training file, named `train.csv`, inside the folder
`init_checkpoint` | PyTorch initial checkpoint for fine-tuning
`save_model_dir` | The folder to save the fine-tuned model. The model will have several version based on the `num_train_epochs`

**Another Parameter**

You can use the default parameters stated on the PyTorch BERT example.

+ ``num_train_epochs`` can be 3.0, 5.0, or 6.0.
+ ``layers`` indicates list of layers which will be taken as feature for classification.
-2 means use pooled output, -1 means concat all layer, the command above means concat
layer-10 and layer-11 (last two layers).
+ ``trunc_medium`` indicates dealing with long texts. -2 means head-only, -1 means tail-only,
0 means head-half + tail-half (e.g.: head256+tail256),
other natural number k means head-k + tail-rest (e.g.: head-k + tail-(512-k)).
+ ``layer_learning_rate`` indicates layer-wise decreasing layer rate.

## Mutant Generation

Our framework, **BiasFinder**, can be instantiated to identify different kinds of bias. In this work, we show how BiasFinder can be instantiated to uncover bias in three different demographic characteristics: gender, occupation, and country-oforigin.

BiasFinder automatically identifies and curates suitable texts in a large corpus of reviews, and transforms these texts into templates. Each template can be used to produce a large number of mutant texts, by filling in placeholders with concrete values associated with a class (e.g., male vs. female) given a demographic characteristic (e.g., gender)(See Section III and IV). Using these mutant texts, **BiasFinder** then runs the SA system under test, checking if it predicts the same sentiment for two mutants associated with a different class (e.g. male vs. female) of the given characteristic (e.g. gender). A pair of such mutants are related through a metamorphic relation where they share the same predicted sentiment from a fair SA system (See Section V and VI).

### 1) Gender Bias
Run this command inside the `codes/gender/` folder

```
python main.py
```

This code will generate mutant texts for gender and saved the mutant texts inside a folder `data/biasfinder/gender/`

### 2) Occupation Bias

Run this command inside the `codes/occupation/` folder

```
python main.py
```

This code will generate mutant texts for occupation and saved the mutant texts inside a folder `data/biasfinder/occupation/`. **Important note:** Occupation bias need StanfordCoreNLP to detect occupation term in the text. Thus please make sure to serve StanfordCoreNLP as an API - [Stackoverflow Guide to serve StanfordCoreNLP as an API](https://stackoverflow.com/questions/32879532/stanford-nlp-for-python).

### 3) Country-of-origin Bias

Run this command inside the `codes/country/` folder

```
python main.py
```

This code will generate mutant texts for country-of-origin and saved the mutant texts inside a folder `data/biasfinder/country/`

## Predict The Mutant Texts using Fine-tuned BERT

### 1) Gender Bias
Run this command inside the `codes/fine-tuning/` folder

```
python predict.py \
  --task_name binary \
  --do_lower_case \
  --eval_data_dir ./../../data/biasfinder/gender/ \
  --vocab_file ./../../models/uncased_L-12_H-768_A-12/vocab.txt \
  --bert_config_file ./../../models/uncased_L-12_H-768_A-12/bert_config.json \
  --seed 42 \
  --layers 11 10 \
  --trunc_medium -1 \
  --init_checkpoint ./../../models/fine-tuning/pytorch_imdb_fine_tuned/epoch5.pt \
  --output_dir ./../../result/biasfinder/gender/
```

This code will produce a prediction of mutant texts inside the folder `result/biasfinder/gender/`

### 2) Occupation Bias

Run this command inside the `codes/fine-tuning/` folder

```
python predict.py \
  --task_name binary \
  --do_lower_case \
  --eval_data_dir ./../../data/biasfinder/occupation/ \
  --vocab_file ./../../models/uncased_L-12_H-768_A-12/vocab.txt \
  --bert_config_file ./../../models/uncased_L-12_H-768_A-12/bert_config.json \
  --seed 42 \
  --layers 11 10 \
  --trunc_medium -1 \
  --init_checkpoint ./../../models/fine-tuning/pytorch_imdb_fine_tuned/epoch5.pt \
  --output_dir ./../../result/biasfinder/occupation/

```

This code will produce a prediction of mutant texts inside the folder `result/biasfinder/occupation/`

### 3) Country-of-origin Bias

Run this command inside the `codes/fine-tuning/` folder

```
python predict.py \
  --task_name binary \
  --do_lower_case \
  --eval_data_dir ./../../data/biasfinder/country/ \
  --vocab_file ./../../models/uncased_L-12_H-768_A-12/vocab.txt \
  --bert_config_file ./../../models/uncased_L-12_H-768_A-12/bert_config.json \
  --seed 42 \
  --layers 11 10 \
  --trunc_medium -1 \
  --init_checkpoint ./../../models/fine-tuning/pytorch_imdb_fine_tuned/epoch5.pt \
  --output_dir ./../../result/biasfinder/country/
```

This code will produce a prediction of mutant texts inside the folder `result/biasfinder/country/`



## Measuring the Bias Uncovering Test Case (BTC)

Mutants of differing classes that are produced from the same template are expected to have the same sentiment. Therefore, if the SA predicts that two mutants of different classes to have different sentiments, they are an evidence of a biased prediction. Such pairs of mutants are output as **bias-uncovering test cases (BTC)**. Thus BTC is a pair that contains 2 different class (e.g. male female for gender bias) and their predictions, such that the Sentiment Analysis produce a different prediction. 
Example of BTC for gender bias: 

`<(male, prediction), (female, prediction)>`

`<(“He is angry”, "positive"), (“She is angry”, "negative")>`

### 1) Gender Bias

Notebook `codes/gender/BTC.ipynb` contains the BTC calculation for gender bias targeting mutant texts.

### 2) Occupation Bias

Notebook `codes/occupation/BTC.ipynb` contains the BTC calculation for occupation bias targeting mutant texts.

### 3) Country-of-origin Bias

Notebook `codes/country/BTC.ipynb` contains the BTC calculation for country-of-origin bias targeting mutant texts.

## Notes
Here the final file structure to better know where to put the models and data:

```
.
|-- LICENSE
|-- README.md
|-- asset
|   |-- gender_associated_word
|   |   |-- masculine-feminine-cleaned.txt
|   |   |-- masculine-feminine-person.txt
|   |   `-- masculine-feminine.txt
|   |-- gender_computer
|   |   |-- female_names_only.csv
|   |   |-- male_names_only.csv
|   |   |-- unique_female_names_and_country.csv
|   |   `-- unique_male_names_and_country.csv
|   |-- imdb
|   |   |-- test.csv
|   |   `-- train.csv
|   |-- imdb_small
|   |   |-- test.csv
|   |   `-- train.csv
|   `-- predefined_occupation_list
|       `-- neutral-occupation.csv 
|-- codes
|   |-- country
|   |   |-- BTC.ipynb
|   |   |-- Coreference.py
|   |   |-- CountryMutantGeneration.py
|   |   |-- Entity.py
|   |   |-- Phrase.py
|   |   |-- main.py
|   |   |-- mutant-generation-example-for-testing.ipynb
|   |   `-- utils.py
|   |-- occupation
|   |   |-- BTC.ipynb
|   |   |-- Coreference.py
|   |   |-- CountryMutantGeneration.py
|   |   |-- Entity.py
|   |   |-- Phrase.py
|   |   |-- main.py
|   |   |-- mutant-generation-example-for-testing.ipynb
|   |   `-- utils.py
|   `-- gender
|       |-- BTC.ipynb
|       |-- Coreference.py
|       |-- MutantGeneration.py
|       |-- Entity.py
|       |-- Phrase.py
|       |-- main.py
|       |-- mutant-generation-example-for-testing.ipynb
|       `-- utils.py
|-- data
|   `-- biasfinder
|       |-- country
|       |   `-- test.csv
|       |-- gender
|       |   `-- test.csv
|       `-- occupation
|           `-- test.csv
|-- models
|   |-- fine-tuning
|   |   |-- pytorch_bert_base_model.bin
|   |   `-- pytorch_imdb_fine_tuned
|   |       |-- epoch1.pt
|   |       |-- epoch2.pt
|   |       |-- epoch3.pt
|   |       |-- epoch4.pt
|   |       `-- epoch5.pt
|   `-- uncased_L-12_H-768_A-12
|       |-- bert_config.json
|       |-- bert_model.ckpt.data-00000-of-00001
|       |-- bert_model.ckpt.index
|       |-- bert_model.ckpt.meta
|       `-- vocab.txt
`-- result
    `-- biasfinder
        |-- country
        |   |-- eval_data_results.txt
        |   `-- results_data.txt
        |-- gender
        |   |-- eval_data_results.txt
        |   `-- results_data.txt
        `-- occupation
            |-- eval_data_results.txt
            `-- results_data.txt
    
```
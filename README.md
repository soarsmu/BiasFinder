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
`asset/occupation/list_XXX/` | It contains pre-determined values for occupation 
`asset/occupation/exclude_list XXX/` | It contains pre-determined exclude list occupation **Need to be explained why it's excluded**


### 2) Prepare Google BERT

Please download [BERT-Base, Uncased](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip) model. You will get `uncased_L-12_H-768_A-12` folder. Put it inside `models/`.


### 3) Fine-Tune BERT on SA using IMDB

#### Convert BERT Tensforflow ckpt into PyTorch checkpoint

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
--fine_tune_data_dir | Data for fine-tuning on downstream task, currently IMDB. You need to put `train.csv` inside the folder
--init_checkpoint | PyTorch initial checkpoint for fine-tuning
--save_model_dir | The folder to save the fine-tuned model. The model will have several version based on the `num_train_epochs`

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

### 1. Gender Bias

### 2. Occupation Bias
### 3. Country-of-origin Bias

## Measuring the Bias Uncovering Test Case (BTC)

BTC is a pair contain of male-female and its prediction, such that the Sentiment Analysis produce a different prediction. 
Example of discordant pair: 

`<(male, prediction), (female, prediction)>`

`<(“He is angry”, 1), (“She is angry”, 0)>`

### 1. Gender Bias
### 2. Occupation Bias
### 3. Country-of-origin Bias


## Country-of-origin Bias Experiment
#### Data Preparation
`codes/prepare-data-from-gender-computer.ipynb` -> prepare data from [GenderComputer](https://github.com/tue-mdse/genderComputer/tree/master/nameLists)

`codes/mutant-generation-using-EEC-template.ipynb` -> mutant generation by substituting name from GenderComputer into EEC template

#### Infer the prediction using Generated Mutant
```shell
python infer.py   \
  --task_name binary \
  --do_lower_case \
  --fine_tune_data_1_dir ./../../data/imdb/ \
  --eval_data_dir ./../../data/gc_mutant/ \
  --vocab_file ./../../models/uncased_L-12_H-768_A-12/vocab.txt \
  --bert_config_file ./../../models/uncased_L-12_H-768_A-12/bert_config.json \
  --init_checkpoint ./../../models/pretrained/pytorch_model_len128_imdb.bin \
  --max_seq_length 128   \
  --train_batch_size 24   \
  --learning_rate 2e-5   \
  --num_train_epochs 1   \
  --seed 42   \
  --layers 11 10   \
  --trunc_medium -1 \
  --output_dir ./../../result/gc_mutant/
```

#### Calculate FPED and FNED
`codes/FNED-FPED-Country.ipynb` -> FNED and FPED calculation for country. But it's easily adopted to use calculation for others, such as gender and occupation.


#### Calculate Number of Discordant Pairs
`codes/discordant-pairs-Country.ipynb` -> Discordant pairs calculation for country. But it's easily adopted to use calculation for others, such as gender and occupation.


## Notes
Here the file structure to better know where to put the models and data:

```
.
|-- LICENSE
|-- README.md
|-- codes
|   |-- FNED-FPED.ipynb
|   |-- discordant-pairs.ipynb
|   |-- fine-tuning
|   |   |-- fairness_test.py
|   |   |-- infer.py
|   |   `-- tokenization.py
|   |-- further-pre-training
|   |   |-- run_pretraining.py
|   |-- mutant-generation.ipynb
|   |-- prepare-data-for-eec.ipynb
|   |-- prepare-data-for-imdb.ipynb
|   `-- prepare-masculine-feminine-word.ipynb
|-- data
|   |-- asset
|   |   |-- masculine-feminine-cleaned.txt
|   |   `-- masculine-feminine.txt
|   |-- eec
|   |   |-- 5persen
|   |   |   |-- female
|   |   |   |   `-- test.csv
|   |   |   |-- male
|   |   |   |   `-- test.csv
|   |   |   |-- test.csv
|   |   |   `-- train.csv
|   |   |-- 6from7
|   |   |   |-- female
|   |   |   |   `-- test.csv
|   |   |   |-- male
|   |   |   |   `-- test.csv
|   |   |   `-- train.csv
|   |   |-- data.csv
|   |   |-- test.csv
|   |   `-- train.csv
|   |-- imdb
|   |   |-- test.csv
|   |   `-- train.csv
|   |-- imdb_mutant
|   |   |-- female
|   |   |   `-- test.csv
|   |   `-- male
|   |       `-- test.csv
|   `-- imdb_small
|       |-- test.csv
|       `-- train.csv
|-- models
|   |-- pretrained
|   |   `-- pytorch_model_len128_imdb.bin
|   `-- uncased_L-12_H-768_A-12
|       |-- bert_config.json
|       |-- bert_model.ckpt.data-00000-of-00001
|       |-- bert_model.ckpt.index
|       |-- bert_model.ckpt.meta
|       `-- vocab.txt
`-- result
    `-- trial_small_imdb
        |-- discordant-pairs.csv
        |-- eval_data_female_results.txt
        |-- eval_data_male_results.txt
        |-- fped-fned-discordant-pairs.txt
        |-- results_data_female.txt
        `-- results_data_male.txt
```
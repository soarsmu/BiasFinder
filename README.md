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

## Setup and Trial for the Experiment

This part will tell you the preparation needed. If you don't find any error in this step. You can go to the [end-to-end program](#end-to-end-program). Then read the [Important Resource](#important-resource) and [Technical Report](#technical-report)

### 1) Prepare the dataset and pretrained model:

#### The Datasets

We use IMDB movie review dataset downloaded from [URL](https://drive.google.com/drive/u/0/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M) proposed by [Zhang et al. (2015)](https://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classification.pdf). The IMDB dataset is saved at `asset/imdb/`.


### 2) Prepare Google BERT:

Please download [BERT-Base, Uncased](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip) model. You will get `uncased_L-12_H-768_A-12` folder. Put it inside `models/`.


<!-- #### Pretrained Model

Pretrained models are available at [here](https://drive.google.com/drive/folders/1Rbi0tnvsQrsHvT_353pMdIbRwDlLhfwM). Currently, I use `pytorch_model_len128_imdb.bin`. Put it inside the folder `models/pretrained/`. -->



### 4) Fine-Tuning using IMDB

#### Before fine-tuning please take a look at several notebook

notebook | description
------------ | -------------
`codes/prepare-data-for-eec.ipynb` | Experiment on EEC data
`codes/prepare-masculine-feminine-word.ipynb` | Litte Experiment on masculine and feminine word
`codes/mutant-generation.ipynb` | **[main part]** Experiment on generating the training and test data

If you don't have a problem for all notebooks, you can continue to the fine-tuning part.

In `codes/mutant-generation.ipynb`, you need a language grammar tool check. A tool to check grammar error in a sentence. There is an API that is built from the source https://github.com/languagetool-org/languagetool. I use docker wrapped API, provided by https://github.com/silvio/docker-languagetool. I implement restful HTTP with this endpoint `http://10.4.4.55:8010//api/v2/check`. You need to run this [docker](https://github.com/silvio/docker-languagetool) and change the endpoint to the appropiate endpoint.


#### Fine-tuning

Here the command for fine-tuning. Run it from the folder `codes/fine-tuning/`

```shell
python infer.py   \
  --task_name binary \
  --do_lower_case \
  --fine_tune_data_1_dir ./../../data/imdb_small/ \
  --fine_tune_data_2_dir ./../../data/eec/6from7/ \
  --eval_data_dir ./../../data/eec/6from7/male/ \
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
  --output_dir ./../../result/trial_on_eec_male/
```

This approach will fine-tune using `--fine_tune_data_1_dir`, then fine-tune again using `--fine_tune_data_2_dir`. `--fine_tune_data_1_dir` is a required parameter and `--fine_tune_data_2_dir` only an optional parameter.
The model then predict evaluation dataset inside folders `--eval_data_dir`.
The result of prediction will be output at `--output_dir`. If the `--output_dir` is already exist, you must delete the folder first.

#### Important Parameter

parameter | description
------------ | -------------
--fine_tune_data_1_dir | First data for fine-tuning, currently IMDB. You need to put `train.csv` inside the folder
--fine_tune_data_2_dir | Second data for fine-tuning, you can left it if you only need one dataset for fine-tuning. You need to put `train.csv` inside the folder
--eval_data_male_dir | Evaluation for the male data. You need to put file `test.csv` inside the folder
--eval_data_female_dir | Evaluation for the female data. You need to put file `test.csv` inside the folder
--output_dir | The folder to put the evaluation result for the model on male data and female data

#### Another Parameter
You can use the default parameter stated on the example

``num_train_epochs`` can be 3.0, 4.0, or 6.0.
Notes: I only use 1 epoch here. 


``layers`` indicates list of layers which will be taken as feature for classification.
-2 means use pooled output, -1 means concat all layer, the command above means concat
layer-10 and layer-11 (last two layers).

``trunc_medium`` indicates dealing with long texts. -2 means head-only, -1 means tail-only,
0 means head-half + tail-half (e.g.: head256+tail256),
other natural number k means head-k + tail-rest (e.g.: head-k + tail-(512-k)).

``pooling_type`` indicates which feature will be used for classification. `mean` means
mean-pooling for hidden state of the whole sequence, `max` means max-pooling, default means
taking hidden state of `[CLS]` token as features.

``layer_learning_rate`` and ``layer_learning_rate_decay`` in ``run_classifier_discriminative.py``
indicates layer-wise decreasing layer rate (See Section 5.3.4).

### 5) Calculate the FPED, FNED

The theory comes from the [AAA 2018 paper](https://www.aies-conference.com/2018/contents/papers/main/AIES_2018_paper_9.pdf).
The `codes/FNED-FPED.ipynb` contain the implementation for it. Please make sure that you have run the fine tuning to get several evalution file needed. Make sure that you put the dataset in the right folder. Match the `output_dir` parameter in the fine-tuning with `output_dir` variable in the notebook.

### 6) Calculate the Number of Discordant Pairs

Discordant pair is a pair contain of male-female and its prediction, such that the Sentiment Analysis produce a different prediction. 
Example of discordant pair: 

`<(male, prediction), (female, prediction)>`

`<(“He is angry”, 1), (“She is angry”, 0)>`

This notebook `codes/discordant-pairs.ipynb` provide the experiment on its calculation

## End-to-End Program

You need to provide a male test data, and a female test data then pass it to parameter `--eval_data_male_dir` and `--eval_data_male_dir` respectively when running `codes/fine-tuning/fairness_test.py`. You also need to know the `--template_size`. Template size is the number of possible generated mutant from a text. The template size for male and female must equal.

The result will be saved in the `--output_dir`.

#### Fine-tuning using IMDB Small, Test on EEC

```shell
python fairness_test_gender.py   \
  --task_name binary \
  --do_lower_case \
  --fine_tune_data_1_dir ./../../data/imdb_small/ \
  --eval_data_male_dir ./../../data/eec/6from7/male/ \
  --eval_data_female_dir ./../../data/eec/6from7/female/ \
  --template_size 1200 \
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
  --output_dir ./../../result/trial_on_eec_without_fine_tuning_eec/
```

#### Fine-tuning using IMDB Small, Fine-tuning EEC, Test on EEC

```shell
python fairness_test_gender.py   \
  --task_name binary \
  --do_lower_case \
  --fine_tune_data_1_dir ./../../data/imdb_small/ \
  --fine_tune_data_2_dir ./../../data/eec/6from7/ \
  --eval_data_male_dir ./../../data/eec/6from7/male/ \
  --eval_data_female_dir ./../../data/eec/6from7/female/ \
  --template_size 1200 \
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
  --output_dir ./../../result/trial_on_eec/
```

#### Fine-tuning using IMDB Full, Test on IMDB Mutant

```shell
python fairness_test_gender.py   \
  --task_name binary \
  --do_lower_case \
  --fine_tune_data_1_dir ./../../data/imdb/ \
  --eval_data_male_dir ./../../data/imdb_mutant/male/ \
  --eval_data_female_dir ./../../data/imdb_mutant/female/ \
  --template_size 20 \
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
  --output_dir ./../../result/trial_on_imdb/
```

## Plug and Play Framework
If you want to test using your own generated mutant text, it is easy. After the previous step is done, You only need to prepare the mutant text into our csv format. Then put it in your desired folder. Set the correct `--eval_data_male_dir` and `--eval_data_male_dir` that refer to your mutant generated folder.

For example on how to prepare the csv data, please read `codes/prepare-data-for-eec.ipynb` and `codes/prepare-data-for-imdb.ipynb`.

The EEC data can be downloaded from its paper. Put it into `data/eec/data.csv`. Then use the `codes/prepare-data-for-eec.ipynb` to generate test.csv train.csv. You can use the notebook to generate female.csv, male.csv  also.

## Important Resource
#### Dataset, Model and Result of the Experiment
This [link](https://drive.google.com/drive/folders/1rnilNLUXjVhtBuZNz4XKNox-WUuJUFP1?usp=sharing) contain a resource to our dataset, model, and experiment result.

#### Technical Report
The technical report will explain what is happen in each sprint.
* [Technical Report Sprint 1](https://drive.google.com/file/d/1NNqyCDb2wNf-UmBhjhd-GnNKXyppmHwp/view?usp=sharing)
* Technical Report Sprint 2
* Technical Report Sprint 3
* Technical Report Sprint 4
* Technical Report Sprint 5
* [Technical Report Sprint 6](https://drive.google.com/file/d/1djRLaplAGsN9qw2cSI2dlIX5Kg4HlFaL/view?usp=sharing)


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
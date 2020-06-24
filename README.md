# Fairness Bugs in Sentiment Analysis

This repo contain the implementation of ICSE 2021 paper - Fairness Bugs in Sentiment Analysis.

Fine-tune text classification is inspired from https://github.com/xuyige/BERT4doc-Classification


## Requirements

For further pre-training, we borrow some code from Google BERT. Thus, we need:

+ tensorflow==1.1x -> currently I use tensorflow-gpu==1.14
+ spacy
+ pandas
+ numpy
+ fastNLP

For fine-tuning, we borrow some codes from pytorch-pretrained-bert package (now well known as transformers). Thus, we need:

+ torch>=0.4.1,<=1.2.0 -> currently I use torch 1.2.0 with cuda 10.0

For nlp task
+ scikit-learn
+ nltk
+ neuralcoref

## Setup and Trial for the Experiment

This part will tell you the preparation needed. If you don't find any error in this step. You can go to the end-to-end program.

### 1) Prepare the dataset and pretrained model:

#### The Datasets

We use IMDB movie review dataset downloaded from [URL](https://drive.google.com/drive/u/0/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M) proposed by [Zhang et al. (2015)](https://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classification.pdf). The notebook `codes/prepare-data.ipynb` will help you to know how to use the IMDB dataset and make it into train test input for our model.

#### Pretrained Model

Pretrained models are available at [here](https://drive.google.com/drive/folders/1Rbi0tnvsQrsHvT_353pMdIbRwDlLhfwM). Currently, I use `pytorch_model_len128_imdb.bin`. Put it inside the folder `models/pretrained/`.


### 2) Prepare Google BERT:

Download [BERT-Base, Uncased](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip) model. You will get `uncased_L-12_H-768_A-12` folder. Put it inside `models/`.


### 3) Further Pre-Training:

Since we use pretrained model form IMDB, this part is not a necessary. But previously I tried to run it and it worked on AG news dataset. For further pretraining using another model or dataset, please take a look at the [BERT fine-tuning repository](https://github.com/xuyige/BERT4doc-Classification)

### 4) Fine-Tuning using IMDB and EEC Then Evaluate the model on Desired Data

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
python fairness_test.py   \
  --task_name binary \
  --do_lower_case \
  --fine_tune_data_1_dir ./../../data/imdb/ \
  --fine_tune_data_2_dir ./../../data/eec/ \
  --eval_data_male_dir ./../../data/imdb_mutant/male/ \
  --eval_data_female_dir ./../../data/imdb_mutant/female/ \
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
  --output_dir ./../../result/imdb_eec
```

#### Important Parameter

parameter | description
------------ | -------------
--fine_tune_data_1_dir | First data for fine-tuning, currently IMDB
--fine_tune_data_2_dir | Second data for fine-tuning, you can left it if you only need one dataset for finetuning
--eval_data_male_dir | Evaluation for the male data
--eval_data_female_dir | Evaluation for the female data
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
I implement it on `codes/FNED-FPED.ipynb`. Please make sure you have run the fine tuning to get several evalution file needed. And make sure that you put the dataset in the right folder. Match the `output_dir` parameter in the fine-tuning with `output_dir` variable in the notebook.

### 6) Calculate the Number of Discordant Pairs

Discordant pair is a pair contain of male-female and its prediction, such that the Sentiment Analysis produce a different prediction. 
Example of discordant pair: 

`<(male, prediction), (female, prediction)>`

`<(“He is angry”, 1), (“She is angry”, 0)>`

## End-to-End Program for Fairness Test


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
|   |-- further-pre-training
|   |   |-- run_pretraining.py
|   |-- mutant-generation.ipynb
|   |-- prepare-data-for-eec.ipynb
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
|   |   |   `-- train.csv
|   |   |-- 7from8
|   |   |   |-- female
|   |   |   |   `-- test.csv
|   |   |   |-- male
|   |   |   |   `-- test.csv
|   |   |   `-- train.csv
|   |-- imdb
|   |   |-- test.csv
|   |   `-- train.csv
|   `-- imdb_mutant
|       |-- female
|       |   `-- test.csv
|       |-- male
|       |   `-- test.csv
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
    `-- exp_on_imdb
        |-- eval_data_female_results.txt
        |-- eval_data_male_results.txt
        |-- results_data_female.txt
        `-- results_data_male.txt

```

For the EEC data, you can download it from the author. Put it into `data/eec/data.csv`
Then use the `codes/prepare-data-for-eec.ipynb` to generate test.csv train.csv. You can use the notebook to generate female.csv, male.csv  also.

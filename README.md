# Fairness Bugs in Sentiment Analysis




Fine-tune text classification is implemented from https://github.com/xuyige/BERT4doc-Classification

I use to this code to explore on fairness testing in Sentiment Analysis Task


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

## Run the Experiment

### 1) Prepare the dataset and pretrained model:

#### The Datasets

We use IMDB movie review dataset downloaded from [URL](https://drive.google.com/drive/u/0/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M) proposed by [Zhang et al. (2015)](https://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classification.pdf). The notebook `data/prepare-data.ipynb` will help you to know how to use the IMDB dataset and make it into train test input for our model.

#### Pretrained Model

Pretrained models are available at [here](https://drive.google.com/drive/folders/1Rbi0tnvsQrsHvT_353pMdIbRwDlLhfwM). Currently, I use `pytorch_model_len128_imdb.bin`. Put it inside the folder `models/pretrained/`.


### 2) Prepare Google BERT:

Download [BERT-Base, Uncased](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip) model. You will get `uncased_L-12_H-768_A-12` folder. Put it inside `models/`.


### 3) Further Pre-Training:

Since we use pretrained model form IMDB, this part is not a necessary. But previously I tried to run it and it worked on AG news dataset.

#### Generate Further Pre-Training Corpus

Here we use AG's News as example:
```shell
python generate_corpus_agnews.py
```
File ``agnews_corpus_test.txt`` can be found in directory ``./data``.

#### Run Further Pre-Training

```shell
python create_pretraining_data.py \
  --input_file=./AGnews_corpus.txt \
  --output_file=tmp/tf_AGnews.tfrecord \
  --vocab_file=./<path to BERT>/uncased_L-12_H-768_A-12/vocab.txt \
  --do_lower_case=True \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=5
  
python run_pretraining.py \
  --input_file=./tmp/tf_AGnews.tfrecord \
  --output_dir=./uncased_L-12_H-768_A-12_AGnews_pretrain \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=./<path to BERT>/uncased_L-12_H-768_A-12/bert_config.json \
  --init_checkpoint=./<path to BERT>/uncased_L-12_H-768_A-12/bert_model.ckpt \
  --train_batch_size=32 \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --num_train_steps=100000 \
  --num_warmup_steps=10000 \
  --save_checkpoints_steps=10000 \
  --learning_rate=5e-5
```

#### Convert Tensorflow checkpoint to PyTorch checkpoint

Since we use a pretained model, we can left this part. Run this part if you do a pre-train again

```shell
python convert_tf_checkpoint_to_pytorch.py \
  --tf_checkpoint_path ./uncased_L-12_H-768_A-12_AGnews_pretrain/model.ckpt-100000 \
  --bert_config_file ./uncased_L-12_H-768_A-12_AGnews_pretrain/bert_config.json \
  --pytorch_dump_path ./uncased_L-12_H-768_A-12_AGnews_pretrain/pytorch_model.bin
```

### 4) Fine-Tuning

#### Fine-Tuning on downstream tasks

Here the command for fine-tuning

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
  --output_dir ./../../result/imdb_eec   \
  --seed 42   \
  --layers 11 10   \
  --trunc_medium -1
```

where ``num_train_epochs`` can be 3.0, 4.0, or 6.0.
Notes: I only use 1 epoch here. 


``layers`` indicates list of layers which will be taken as feature for classification.
-2 means use pooled output, -1 means concat all layer, the command above means concat
layer-10 and layer-11 (last two layers).

``trunc_medium`` indicates dealing with long texts. -2 means head-only, -1 means tail-only,
0 means head-half + tail-half (e.g.: head256+tail256),
other natural number k means head-k + tail-rest (e.g.: head-k + tail-(512-k)).

There also other arguments for fine-tuning:

``pooling_type`` indicates which feature will be used for classification. `mean` means
mean-pooling for hidden state of the whole sequence, `max` means max-pooling, default means
taking hidden state of `[CLS]` token as features.

``layer_learning_rate`` and ``layer_learning_rate_decay`` in ``run_classifier_discriminative.py``
indicates layer-wise decreasing layer rate (See Section 5.3.4).


## Notes
Here the file structure to better know where to put the models and data:
```
.
|-- LICENSE
|-- README.md
|-- codes
|   |-- FNED-FPED.ipynb
|   |-- fine-tuning
|   |   |-- convert_tf_checkpoint_to_pytorch.py
|   |   |-- extract_features.py
|   |   |-- fairness_test.py
|   |   |-- modeling.py
|   |   |-- modeling_last_concat_avg.py
|   |   |-- modeling_multitask.py
|   |   |-- modeling_single_layer.py
|   |   |-- optimization.py
|   |   |-- run_classifier.py
|   |   |-- run_classifier_discriminative.py
|   |   |-- run_classifier_multitask.py
|   |   |-- run_classifier_no_decay.py
|   |   |-- run_classifier_single_layer.py
|   |   `-- tokenization.py
|   `-- further-pre-training
|       |-- create_pretraining_data.py
|       |-- extract_features.py
|       |-- generate_corpus_agnews.py
|       |-- modeling.py
|       |-- optimization.py
|       |-- run_pretraining.py
|       `-- tokenization.py
|-- data
|   |-- eec
|   |   |-- data.csv
|   |   |-- female.csv
|   |   |-- male.csv
|   |   |-- test.csv
|   |   `-- train.csv
|   |-- imdb
|   |   |-- test.csv
|   |   `-- train.csv
|   `-- prepare-data.ipynb
|-- models
|   |-- pretrained
|   |   `-- pytorch_model_len128_imdb.bin
|   `-- uncased_L-12_H-768_A-12
|       |-- bert_config.json
|       |-- bert_model.ckpt.data-00000-of-00001
|       |-- bert_model.ckpt.index
|       |-- bert_model.ckpt.meta
|       `-- vocab.txt
|-- result
    `-- imdb_eec
        |-- eval_after_data_1_results_ep1.txt
        |-- eval_after_data_2_results_ep1.txt
        |-- eval_before_data_1_results_ep1.txt
        |-- results_after_data_1_ep1.txt
        |-- results_after_data_2_ep1.txt
        |-- results_before_data_1_ep1.txt
        `-- test.csv
```

For the EEC data, you can download it from the author. Put it into `data/eec/data.csv`
Then use the `prepare-data.ipynb` to generate test.csv train.csv. You can use the notebook to generate female.csv, male.csv  also.


### FNED-FPED
The theory comes from the [AAA 2018 paper](https://www.aies-conference.com/2018/contents/papers/main/AIES_2018_paper_9.pdf).
I implement it on `codes/FNED-FPED.ipynb`. Please make sure you have run the fine tuning to get several evalution file needed. And make sure that you put the dataset in the right folder.
## genereate mutants using biasfinder on IMDB test set
## A model which is fine-tuned using IMDB train set 
## This model is used for predicting the mutants
# python test.py --task imdb --model bert-base-uncased --dataset imdb
python test.py --task imdb --model bert-base-cased --dataset imdb
# python test.py --task imdb --model roberta-base --dataset imdb
# python test.py --task imdb --model xlnet-base-cased --dataset imdb
# python test.py --task imdb --model albert-base-v1 --dataset imdb

# python test.py --task imdb --model albert-base-v2 --dataset imdb
# python test.py --task imdb --model microsoft/mpnet-base --dataset imdb
# python test.py --task imdb --model google/electra-base-generator --dataset imdb
# python test.py --task imdb --model facebook/muppet-roberta-base --dataset imdb
# python test.py --task imdb --model microsoft/deberta-base --dataset imdb

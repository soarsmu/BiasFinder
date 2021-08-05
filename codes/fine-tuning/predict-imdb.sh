## genereate mutants using biasfinder on IMDB test set
## A model which is fine-tuned using IMDB train set 
## This model is used for predicting the mutants
python predict.py --mutation-tool biasfinder --bias-type gender --task imdb --model bert-base-uncased --mutant imdb
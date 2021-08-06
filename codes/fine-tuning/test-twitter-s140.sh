## genereate mutants using biasfinder on IMDB test set
## A model which is fine-tuned using IMDB train set 
## This model is used for predicting the mutants

python test.py --task twitter_semeval --model bert-base-uncased --dataset twitter_s140
python test.py --task twitter_semeval --model bert-base-cased --dataset twitter_s140
python test.py --task twitter_semeval --model roberta-base --dataset twitter_s140
python test.py --task twitter_semeval --model xlnet-base-cased --dataset twitter_s140

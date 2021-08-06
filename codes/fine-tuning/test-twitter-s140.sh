## genereate mutants using biasfinder on IMDB test set
## A model which is fine-tuned using IMDB train set 
## This model is used for predicting the mutants

python test.py --task twitter_s140 --model bert-base-uncased --dataset twitter_s140
python test.py --task twitter_s140 --model bert-base-cased --dataset twitter_s140
python test.py --task twitter_s140 --model roberta-base --dataset twitter_s140
# python test.py --task twitter_s140 --model xlnet-base-cased --dataset twitter_s140

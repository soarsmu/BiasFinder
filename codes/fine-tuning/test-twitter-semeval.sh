## genereate mutants using biasfinder on IMDB test set
## A model which is fine-tuned using IMDB train set 
## This model is used for predicting the mutants

# python test.py --task twitter_semeval --model bert-base-uncased --dataset twitter_semeval
# python test.py --task twitter_semeval --model bert-base-cased --dataset twitter_semeval
# python test.py --task twitter_semeval --model roberta-base --dataset twitter_semeval
# python test.py --task twitter_semeval --model xlnet-base-cased --dataset twitter_semeval


python test.py --task twitter_semeval --model albert-base-v2  --dataset twitter_semeval
python test.py --task twitter_semeval --model microsoft/mpnet-base  --dataset twitter_semeval
python test.py --task twitter_semeval --model microsoft/deberta-base  --dataset twitter_semeval
python test.py --task twitter_semeval --model facebook/muppet-roberta-base  --dataset twitter_semeval
python test.py --task twitter_semeval --model google/electra-base-generator  --dataset twitter_semeval
